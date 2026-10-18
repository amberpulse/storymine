#!/usr/bin/env python3
"""storymine 内核 CLI

从人类写的故事里，挖出「叙事动作」——回答写作者那个问题：接下来该发生什么。

命令：
    extract    提取叙事动作入库（逐章处理，可续跑）
    query      卡住了就问它：我写到…… → 返回手法卡
    stats      库统计
    list       浏览手法表
    failures   列出还没解决的失败/可疑章
    delete     删除条目（默认只预览，--yes 才动手）

设计约定：
  · 配置化——端点 / 模型 / 库路径 / 输入 全可分设
  · 逐件处理——一章一章来，成功即下一个
  · 失败按类分流——瞬时的重试；配额墙立刻停下报错；内容审查拒当场判死、不重试；脏数据记下跳过
  · 退出码说真话——有章失败 / 一条没入库 → 非 0

零第三方依赖：只用 stdlib + curl。所有输出 UTF-8。
"""
import argparse
import json
import os
import pathlib
import re
import sqlite3
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = pathlib.Path(__file__).resolve().parent

# Windows：调外部控制台程序（curl 等）必须带 CREATE_NO_WINDOW。
# 否则父进程没有控制台时，Windows 会给每个子进程开一个新的控制台窗口（弹窗）。
_NOWIN = 0x08000000 if os.name == "nt" else 0

# ─────────────────────────── 默认配置 ───────────────────────────
DEFAULTS = {
    "api": {
        "base_url": "https://api.deepseek.com",     # 通用默认；换别的端点用 --base-url 或 config.json
        "model": "deepseek-chat",                   # 任何 OpenAI 兼容的模型都行
        "key_env": "STORYMINE_API_KEY",             # key 走环境变量或 config.json 指的 env 文件
        "env_file": "",             # 留空 = 只认环境变量
        "timeout": 240,
        "temperature": 0.4,
        "retries": 2,
        "max_chars": 8000,          # 单章送入上限，超了截断
    },
    "store": {
        "db": str(ROOT / "storymine.db"),
        "tmp": str(ROOT / ".tmp"),
    },
    "extract": {
        "vocab": "",                # 手法表文件（留空 = 自由命名）
    },
    "query": {
        "top_n": 5,
        "max_entries": 400,         # 塞进 prompt 的条目上限
    },
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, source_path TEXT,
    total_chars INTEGER, created_at TEXT DEFAULT (datetime('now','localtime')));
CREATE TABLE IF NOT EXISTS chapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER, seq INTEGER,
    title TEXT, chars INTEGER,
    UNIQUE(book_id, seq));
CREATE TABLE IF NOT EXISTS beats (
    id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER, chapter_seq INTEGER, idx INTEGER,
    move TEXT, label TEXT, how TEXT, situation TEXT,
    why TEXT, pos TEXT, characters TEXT, raw TEXT,
    created_at TEXT DEFAULT (datetime('now','localtime')));
CREATE INDEX IF NOT EXISTS idx_beats_book_ch ON beats(book_id, chapter_seq);
CREATE INDEX IF NOT EXISTS idx_beats_move ON beats(move);
CREATE INDEX IF NOT EXISTS idx_beats_label ON beats(label);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, book_id INTEGER, chapter_seq INTEGER,
    beats INTEGER, in_tok INTEGER, out_tok INTEGER, seconds REAL, status TEXT,
    note TEXT, ts TEXT DEFAULT (datetime('now','localtime')));
"""

MOVE_TABLE = ["引入", "展开", "升级", "转场", "反转", "揭露", "铺垫",
              "放缓", "悬置", "收束", "关系推进", "立场转变"]

# ─────────────────────────── 异常分类 ───────────────────────────
class QuotaWall(Exception):
    """配额/额度墙——重试没有意义，立刻停。"""

class Transient(Exception):
    """瞬时故障（网络/超时/5xx）——值得重试。"""

class BadOutput(Exception):
    """模型输出不可解析——**偶发**，值得重试一次（实测重跑就过）。"""


class ContentRejected(Exception):
    """渠道内容审查拒——**永久**，重试一百次还是拒，不许自动重试（纯烧钱）。"""

# ─────────────────────────── 配置 ───────────────────────────
def deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None):
    cfg = json.loads(json.dumps(DEFAULTS))
    if path is None:
        cand = ROOT / "config.json"
        path = str(cand) if cand.exists() else None
    if path and pathlib.Path(path).exists():
        raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        cfg = deep_merge(cfg, raw)
    # 环境变量兜底（可分设）
    cfg["api"]["base_url"] = os.environ.get("STORYMINE_BASE_URL", cfg["api"]["base_url"])
    cfg["api"]["model"] = os.environ.get("STORYMINE_MODEL", cfg["api"]["model"])
    return cfg


def load_key(cfg):
    """key 只有两种来源：环境变量，或 config.json 指向的 env 文件。代码里不留任何路径。"""
    name = cfg["api"]["key_env"]
    if os.environ.get(name):
        return os.environ[name]
    env_file = cfg["api"].get("env_file") or ""
    if env_file and pathlib.Path(env_file).exists():
        for line in pathlib.Path(env_file).read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(
        f"没找到 API key（{name}）。二选一：\n"
        f'  ① 环境变量：export {name}=xxx\n'
        f'  ② 项目根 config.json：{{"api": {{"env_file": "<含 {name}=xxx 的文件路径>"}}}}\n'
        f"（key 只走这两条路，不写进代码——开源版靠这个保持干净）")


# ─────────────────────────── LLM 调用 ───────────────────────────
def call_llm(prompt, cfg, temperature=None):
    """curl 直连（本机代理反而会挂，别加 -x）。payload 走文件，中文才不乱。"""
    tmp = pathlib.Path(cfg["store"]["tmp"])
    tmp.mkdir(parents=True, exist_ok=True)
    pf = tmp / "_payload.json"
    payload = {
        "model": cfg["api"]["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg["api"]["temperature"] if temperature is None else temperature,
    }
    pf.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    last = None
    for attempt in range(cfg["api"]["retries"] + 1):
        r = subprocess.run(
            ["curl", "-s", "-w", "\n@@HTTP:%{http_code}", "--max-time",
             str(cfg["api"]["timeout"]),
             "-X", "POST", cfg["api"]["base_url"].rstrip("/") + "/chat/completions",
             "-H", "Content-Type: application/json",
             "-H", "Authorization: Bearer " + load_key(cfg),
             "--data-binary", "@" + str(pf)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            creationflags=_NOWIN, stdin=subprocess.DEVNULL)
        body, _, code = (r.stdout or "").rpartition("\n@@HTTP:")
        code = (code or "").strip()
        low = (body or "").lower()

        if code == "429" or "insufficient credits" in low or "quota" in low or "rate limit" in low:
            raise QuotaWall(f"HTTP {code} {body[:200]}")
        if not code or code.startswith("5") or code == "000" or not body.strip():
            last = Transient(f"HTTP {code} {body[:150]}")
        else:
            try:
                data = json.loads(body)
            except Exception:
                last = BadOutput("响应不是 JSON：" + body[:150])
            else:
                if "error" in data:
                    msg = json.dumps(data["error"], ensure_ascii=False)[:300]
                    m = msg.lower()
                    if "credit" in m or "quota" in m:
                        raise QuotaWall(msg)
                    if any(k in msg for k in ("DataInspectionFailed", "inappropriate content",
                                              "content_policy", "content filter")) \
                            or "内容审核" in msg:
                        raise ContentRejected(msg)      # 永久性，不重试
                    last = BadOutput(msg)
                else:
                    usage = data.get("usage", {})
                    return data["choices"][0]["message"]["content"], usage
        if attempt < cfg["api"]["retries"]:
            time.sleep(3 * (attempt + 1))
    raise last if last else RuntimeError("unknown failure")


def parse_json_array(text):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    i, j = t.find("["), t.rfind("]")
    if i < 0 or j < 0:
        raise BadOutput("没有 JSON 数组：" + t[:150])
    try:
        return json.loads(t[i:j + 1])
    except Exception as e:
        # 统一成 BadOutput——否则 JSONDecodeError 会从重试逻辑里漏出去
        raise BadOutput(f"JSON 解析失败：{str(e)[:120]}")


# ─────────────────────────── 分章 ───────────────────────────
# 语料库实测四种章标题格式：
#   ① 第1章xx   ② 第一章xx   ③ 第一部X 第一章yy（部名与章名同行）   ④ 001xx / 1、xx
_CH_NUM = r"[0-9０-９一二三四五六七八九十百千万两零〇○]"
_PAT_STD = re.compile(rf"^[\s\u3000]*(第\s*{_CH_NUM}+\s*[章回节][^\n]*)$", re.M)
_PAT_LOOSE = re.compile(rf"^[\s\u3000]*[^\n]{{1,12}}?第\s*{_CH_NUM}+\s*[章回节][^\n]*$", re.M)
_PAT_NUMLINE = re.compile(r"^[\s\u3000]*([0-9]{1,5})\s*[、.．,，:：]?\s*(\S[^\n]{0,28})$", re.M)


def _build_chapters(text, marks):
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((m.group(0).strip(), text[m.end():end].strip()))
    return out


def _rising(marks, ratio=0.75):
    """纯数字标题没有"第…章"锚点，靠'编号递增'把正文里的数字行滤掉。"""
    if len(marks) < 4:
        return False
    nums = [int(m.group(1)) for m in marks]
    inc = sum(1 for a, b in zip(nums, nums[1:]) if b > a)
    return inc / (len(nums) - 1) >= ratio


_CN_D = {"零": 0, "〇": 0, "○": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
         "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def cn2num(s):
    if s.isdigit():
        return int(s)
    total = section = num = 0
    for ch in s:
        if ch in _CN_D:
            num = _CN_D[ch]
        elif ch == "十":
            section += (num or 1) * 10
            num = 0
        elif ch == "百":
            section += (num or 1) * 100
            num = 0
        elif ch == "千":
            section += (num or 1) * 1000
            num = 0
        elif ch == "万":
            total += (section + num) * 10000
            section = num = 0
    return total + section + num


def chapter_no(title):
    m = re.search(rf"第\s*([0-9]+|{_CH_NUM}+)\s*[章回节]", title)
    if m:
        return cn2num(m.group(1))
    m2 = re.match(r"^\s*([0-9]{1,5})", title)
    return int(m2.group(1)) if m2 else None


def _tidy(chapters):
    """语料库两个通病，都会把章数灌成两倍、内容灌成空壳：

    ① 开头的目录页——连续多"章"正文短得不像话（几十字一行的章名清单）
    ② 标题重复——同一章号接连出现两次（目录残留 / 双版本混排）。
       注意：正文挂在其后**那**个标题上，所以不能简单丢后者——
       两个标题行常常是紧挨着的（前者正文是空的），丢错就把整本书灌成空壳。
    """
    i = 0
    while i < len(chapters) and len(chapters[i][1]) < 30:
        i += 1
    if i >= 4:
        chapters = chapters[i:]
    out = []
    for title, body in chapters:
        no = chapter_no(title)
        if out and no is not None and chapter_no(out[-1][0]) == no:
            if len(body) > len(out[-1][1]):        # 合并：留长的正文，标题用前一个
                out[-1] = (out[-1][0], body)
            continue
        out.append((title, body))
    return out


def split_chapters(text, pattern=None):
    """自适应分章。严格不行就放松，再不行试纯数字标题。

    pattern 给了就**只用它**——这是"跟用户商量后重新切分"的入口。
    """
    if pattern:
        pat = re.compile(pattern, re.M)
        marks = list(pat.finditer(text))
        return _tidy(_build_chapters(text, marks)) if len(marks) >= 2 else []
    marks = list(_PAT_STD.finditer(text))
    if len(marks) >= 4:
        return _tidy(_build_chapters(text, marks))
    marks = list(_PAT_LOOSE.finditer(text))
    if len(marks) >= 4:
        return _tidy(_build_chapters(text, marks))
    cand = list(_PAT_NUMLINE.finditer(text))
    if _rising(cand):
        return _tidy(_build_chapters(text, cand))
    return []


def diagnose_split(chapters, text):
    """切分体检——返回问题列表（空 = 健康）。

    宁可误报，不可静默：切歪了却不说，用户灌一百本都不知道有几本没进去。
    尤其：**用户自己给的正则很容易把正文行也吃进来**（章数暴涨），
    这时"章均字数"看着完全正常，只有"章号重复率"能抓出来。
    """
    issues = []
    if not chapters:
        issues.append("一章都没切出来")
        return issues
    if len(chapters) < 3:
        issues.append(f"只切出 {len(chapters)} 章")
    lens = [len(b) for _, b in chapters]
    avg = sum(lens) / len(lens)
    if avg < 100:
        issues.append(f"章均仅 {avg:.0f} 字 —— 疑似把正文行当成了章标题")
    if len(chapters) >= 20:
        short = sum(1 for b in lens if b < 200) / len(lens)
        if short > 0.15:
            issues.append(f"{short*100:.0f}% 的章不足 200 字 —— 疑似把正文行也当成章标题了（切多了）")
    return issues


# ─────────────────────────── 提取 prompt ───────────────────────────
PROMPT_TMPL = """你是一个故事节拍分析器。下面是一章中文小说。

你的任务**不是复述剧情**，而是分析：**作者在这一章里，用了哪些叙事动作来推动故事**。
一句话：不要告诉我"发生了什么"，告诉我"作者在这里干了什么"。

把这一章切成 3-6 个环节。每个环节输出：

1. situation：写到这里，故事**面对的局面**是什么？（35 字以内）
   - 必须抽象：**不许出现角色名字**，用"主角 / 陌生人 / 房客 / 追兵"这类角色功能称呼。
   - 写"处境"，不是写"发生了什么"。
   - ✔ "落魄的普通人，需要给生活开一个出口"
   - ✘ "郝仁找不到工作，去公园睡了一觉"

2. move：作者在这里用的叙事动作，从下面选**一个**：
   {moves}

3. label：用 4-6 个字给这个动作起个名字。**要求朴素直白，像句大白话**——不要文学化、不要章回体。
   - ✔「把人引上门」「先说半句」「露个破绽」「换个场子」「先垫一笔」
   - ✘「命运交汇」「时空错位」「异类现形」（太文绉绉，像章回标题）
{vocab_block}
4. how：作者用的是**什么手法**（35 字以内）
   - **关键：停在"手法"这一层，不要下探到具体桥段。**
   - 检验标准：这句话要能**原样套到另一个故事里**——换个角色、换个背景，照样成立，才算合格。
   - ✔「用一个不合常理的日常细节，暴露新角色的异常」（放哪本书都能用）
   - ✔「让新角色带着一条与主角本人有关的信息出现」
   - ✔「借环境变化暗示时间流逝，把静止的主角推出来」
   - ✘「用重物轻提的反差细节打破认知」（"重物轻提"是这本书自己的桥段，换本书就不成立）
   - ✘「让陌生人递上写有特定地点的凭证」（等于把原文复述一遍）

5. why：作者这么做是为了什么（20 字以内。例：制造悬念 / 缓解张力 / 建立信任 / 放大不安 / 埋钩子 / 给读者喘口气）

6. pos：这个环节在章的什么位置（开头 / 中段 / 结尾）

7. characters：涉及的角色名（原文里的名字，用于回溯）

8. raw：对应的原文片段（不超过 80 字，**逐字摘录**，用于回溯）

输出 JSON 数组，不要 markdown，不要解释。

章节文本：
---
{chapter}
---"""


def build_prompt(body, vocab=None):
    vocab_block = ""
    if vocab:
        vocab_block = ("3.5 **优先从这个手法表里挑 label**；表里确实没有合适的，才自己起一个：\n"
                       + "   " + " / ".join(vocab) + "\n")
    return PROMPT_TMPL.format(moves=" / ".join(MOVE_TABLE), vocab_block=vocab_block,
                              chapter=body)


def load_vocab(cfg):
    p = cfg["extract"].get("vocab") or ""
    if not p or not pathlib.Path(p).exists():
        return None
    txt = pathlib.Path(p).read_text(encoding="utf-8")
    return [w.strip() for w in re.split(r"[\n,，、/]+", txt) if w.strip()]


# ─────────────────────────── 原文回溯 ───────────────────────────
def raw_hit(raw, body):
    """模型爱用「……」拼接不相邻的原文——按省略号切段逐段匹配。"""
    parts = [p.strip() for p in re.split(r"[⋯…]{1,}|\.{3,}", raw or "") if len(p.strip()) >= 4]
    if not parts:
        parts = [(raw or "").strip()]
    got = sum(1 for p in parts if p in body)
    return got, len(parts)


# ─────────────────────────── 入库 ───────────────────────────
def open_db(cfg):
    db = pathlib.Path(cfg["store"]["db"])
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.executescript(SCHEMA)
    return conn


def book_title_from_path(path):
    """'315_道起皇城(1-78章).txt' → '道起皇城'（语料库命名带序号和章节范围）"""
    name = pathlib.Path(path).stem
    m = re.match(r"^\d+_(.+?)(?:\(\d+-\d+章\))?$", name)
    return (m.group(1) if m else name).strip()


def get_or_create_book(conn, path):
    p = pathlib.Path(path)
    row = conn.execute("SELECT id FROM books WHERE source_path=?", (str(p),)).fetchone()
    if row:
        return row[0]
    size = p.stat().st_size
    cur = conn.execute("INSERT INTO books (title, source_path, total_chars) VALUES (?,?,?)",
                       (book_title_from_path(p), str(p), size))
    conn.commit()
    return cur.lastrowid


# ─────────────────────────── 命令：extract ───────────────────────────
def cmd_extract(args, cfg):
    cfg = deep_merge(cfg, {})
    if args.db:
        cfg["store"]["db"] = args.db
    if args.model:
        cfg["api"]["model"] = args.model
    if args.base_url:
        cfg["api"]["base_url"] = args.base_url

    targets = []
    src = pathlib.Path(args.input)
    if src.is_dir():
        targets = sorted([p for p in src.glob("*.txt")], key=lambda p: p.name)
        print(f"[扫描] {src} → {len(targets)} 个 txt")
    else:
        targets = [src]

    vocab = load_vocab(cfg)
    if vocab:
        print(f"[手法表] 已载入 {len(vocab)} 个手法名")

    a, _, b = (args.chapters or "1-999999").partition("-")
    lo, hi = int(a), int(b or a)

    conn = open_db(cfg)
    grand = 0
    n_fail = 0
    for ti, book in enumerate(targets, 1):
        if not book.exists():
            print(f"[{ti}/{len(targets)}] {book.name} —— 文件不存在，跳过")
            continue
        text = book.read_text(encoding="utf-8", errors="replace")
        bid = get_or_create_book(conn, book)
        chapters = split_chapters(text, args.pattern)
        whole_taken = False
        if not chapters:
            # 切不出章 ≠ 不能用：整篇当一章收起。
            # 这是"用户丢一个片段/短篇/单章文件"的正门——以前这里直接弹回，
            # 逼得 agent 自己写脚本绕道调内核函数（绕过了所有防护）。
            chapters = [(book_title_from_path(book), text)]
            whole_taken = True
            issues = ([f"切不出章标题，原文却有 {len(text)//10000} 万字 —— 整篇当一章只会处理开头，"
                       f"建议 --pattern 重切"] if len(text) > 50000 else [])
        else:
            issues = diagnose_split(chapters, text)

        if whole_taken and not issues:
            print(f"   ℹ️ 《{book_title_from_path(book)}》没分出章标题 → "
                  f"按「整篇一章」收起（{len(text)} 字）")

        if issues:
            conn.execute("""INSERT INTO runs (book_id,chapter_seq,beats,status,note)
                            VALUES (?,?,?,?,?)""",
                         (bid, 0, 0, "nosplit", " / ".join(issues)))
            conn.commit()
            if not args.skip:
                print(f"\n  {'='*62}")
                print(f"  ⚠️  切分可疑：《{book_title_from_path(book)}》")
                for it in issues:
                    print(f"        · {it}")
                print(f"        文件：{book}")
                print(f"        开头：{text[:100]!r}")
                print(f"        ── 怎么办 ──")
                print(f"        ① 重切：加 --pattern \"<章标题正则>\"（跟用户商量定了规则再跑）")
                print(f"        ② 忽略：加 --skip，以后不再报这一本")
                print(f"  {'='*62}")
            print(f"        （仍按当前切法入库 {len(chapters)} 章，可事后重切）")
        print(f"\n[{ti}/{len(targets)}] 《{book_title_from_path(book)}》 {len(chapters)} 章")
        n_ok = 0
        for seq, (title, body) in enumerate(chapters, 1):
            if seq < lo or seq > hi:
                continue
            done = conn.execute("SELECT COUNT(*) FROM beats WHERE book_id=? AND chapter_seq=?",
                                (bid, seq)).fetchone()[0]
            if done and not args.force:
                print(f"   [{seq}] {title[:24]:<26} 跳过（已入库 {done} 条）")
                continue
            if args.force:
                conn.execute("DELETE FROM beats WHERE book_id=? AND chapter_seq=?", (bid, seq))
                conn.commit()

            t0 = time.time()
            status, note, n = "ok", "", 0
            usage = {}
            print(f"   [{seq}] {title[:24]:<26} {len(body)}字 ...", end="", flush=True)
            try:
                # 坏 JSON 是**偶发**的（实测重跑就过）→ 值得重试一次。
                # 内容审查拒 / 额度墙 / 瞬时故障在 call_llm 里就按类抛出了，不会走到这里。
                prompt = build_prompt(body[:cfg["api"]["max_chars"]], vocab)
                for _try in range(2):
                    out, usage = call_llm(prompt, cfg, temperature=args.temperature)
                    try:
                        beats = parse_json_array(out)
                        break
                    except BadOutput:
                        if _try:
                            raise
                hit = tot = 0
                for i, bt in enumerate(beats, 1):
                    r = bt.get("raw", "")
                    h, t = raw_hit(r, body)
                    hit += h
                    tot += t
                    conn.execute(
                        """INSERT INTO beats (book_id,chapter_seq,idx,move,label,how,situation,
                           why,pos,characters,raw) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (bid, seq, i, bt.get("move", ""), bt.get("label", ""), bt.get("how", ""),
                         bt.get("situation", ""), bt.get("why", ""), bt.get("pos", ""),
                         json.dumps(bt.get("characters", []), ensure_ascii=False), r))
                conn.execute("DELETE FROM chapters WHERE book_id=? AND seq=?", (bid, seq))
                conn.execute("INSERT INTO chapters (book_id,seq,title,chars) VALUES (?,?,?,?)",
                             (bid, seq, title, len(body)))
                n = len(beats)
                note = f"回溯 {hit}/{tot}"
            except QuotaWall as e:
                conn.execute("""INSERT INTO runs (book_id,chapter_seq,beats,status,note)
                                VALUES (?,?,?,?,?)""", (bid, seq, 0, "quota", str(e)[:300]))
                conn.commit()
                print(f"\n\n  ✖ 撞上配额/额度墙，**立刻停下**（重试没意义）：{str(e)[:160]}")
                print("    已入库的章都保住了，充上额度后原命令重跑即可续。")
                conn.close()
                raise SystemExit(3)
            except (Transient, BadOutput, ContentRejected) as e:
                status = {"Transient": "transient", "BadOutput": "badoutput",
                          "ContentRejected": "rejected"}.get(type(e).__name__, "error")
                note = str(e)[:300]
            except Exception as e:
                status = "error"
                note = f"{type(e).__name__}: {str(e)[:250]}"

            itok = usage.get("prompt_tokens", 0) or 0
            otok = usage.get("completion_tokens", 0) or 0
            if n:
                conn.execute("""INSERT INTO runs (book_id,chapter_seq,beats,status,note,in_tok,out_tok,seconds)
                                VALUES (?,?,?,?,?,?,?,?)""",
                             (bid, seq, n, status, note, itok, otok, time.time() - t0))
                conn.commit()
                n_ok += n
                print(f" {n}条 ({note}) 进{itok}/出{otok} {time.time()-t0:.0f}s")
            else:
                conn.execute("""INSERT INTO runs (book_id,chapter_seq,beats,status,note,in_tok,out_tok,seconds)
                                VALUES (?,?,?,?,?,?,?,?)""",
                             (bid, seq, 0, status, note, itok, otok, time.time() - t0))
                conn.commit()
                n_fail += 1
                print(f" ✖ 失败[{status}] {note[:120]}")
        print(f"   —— 《{book_title_from_path(book)}》 本轮入库 {n_ok} 条")
        grand += n_ok
    conn.close()
    print(f"\n完成：本轮共入库 {grand} 条 → {cfg['store']['db']}")
    if n_fail:
        print(f"\n⚠️  有 {n_fail} 章失败（**退出码非 0**）——跑 `storymine.py failures` 看是哪几章。")
        print(f"    瞬时的（坏 JSON）原命令重跑即可；内容审查拒的重跑也没用，得换料。")
    return n_fail


# ─────────────────────────── 命令：query ───────────────────────────
PROMPT_QUERY = """一个写作者卡住了，他的原话：

「{q}」

下面是一个"人类作者叙事手法库"的条目。请从中挑出**最相关的 {n} 条**，按相关度排序。

对每条说明：**这一招能帮他解决什么**（一句话，别复述条目内容）。

输出 JSON 数组：
[{{"id":"编号", "why_usable":"…"}}]

只输出 JSON，不要解释。

手法库：
{lib}"""


def _llm_pick(rows, q, n, cfg):
    """让模型从这一批条目里挑 n 条 → [{"id":..,"why_usable":..}, ...]"""
    lib = "\n".join(
        f"[{r[0]}] 动作:{r[3]} | 手法:{r[4]} | 局面:{r[6]} | 做法:{r[5]}" for r in rows)
    out, _ = call_llm(PROMPT_QUERY.format(q=q, n=n, lib=lib), cfg, temperature=0.3)
    try:
        return parse_json_array(out)
    except Exception:
        picks = []
        for seg in re.findall(r"\{[^{}]*\}", out):
            try:
                picks.append(json.loads(seg))
            except Exception:
                pass
        return picks


def cmd_query(args, cfg):
    if args.db:
        cfg["store"]["db"] = args.db
    if args.model:
        cfg["api"]["model"] = args.model
    if args.base_url:
        cfg["api"]["base_url"] = args.base_url

    conn = open_db(cfg)   # 统一走 open_db：它会自动建表（空库上 stats/list/failures/query 都不该崩）
    rows = conn.execute(
        """SELECT b.id, k.title, b.chapter_seq, b.move, b.label, b.how, b.situation
           FROM beats b LEFT JOIN books k ON k.id=b.book_id
           ORDER BY b.id""").fetchall()
    if not rows:
        raise SystemExit(f"库是空的：{cfg['store']['db']}")

    cap = cfg["query"]["max_entries"]
    n = args.n or cfg["query"]["top_n"]
    idx = {r[0]: r for r in rows}

    # 分批覆盖全库——**不能只取最后 cap 条**：库一大，前面的老条目就永远进不了 prompt
    batches = [rows[i:i + cap] for i in range(0, len(rows), cap)]
    if len(batches) > 1:
        print(f"[扫描] 全库 {len(rows)} 条 → 分 {len(batches)} 批", flush=True)

    cands, seen = [], set()          # [(row_id, why_usable)]
    for bi, batch in enumerate(batches, 1):
        if len(batches) > 1:
            print(f"   第 {bi}/{len(batches)} 批（{len(batch)} 条）…", end="", flush=True)
        got = 0
        for p in _llm_pick(batch, args.text, n, cfg):
            if not isinstance(p, dict):
                continue
            try:
                pid = int(p.get("id"))
            except (TypeError, ValueError):
                continue
            if pid in idx and pid not in seen:
                seen.add(pid)
                cands.append((pid, p.get("why_usable", "")))
                got += 1
        if len(batches) > 1:
            print(f" 得 {got} 条", flush=True)

    if not cands:
        print("（模型一条都没挑出来——换个说法再试，或者库太偏）")
        return

    # 批次多了要先精选，否则排前的批次会占满名额
    if len(batches) > 1 and len(cands) > n:
        print(f"[精选] 从 {len(cands)} 条候选里挑 {n} 条 …", flush=True)
        picked = []
        for p in _llm_pick([idx[pid] for pid, _ in cands], args.text, n, cfg):
            if not isinstance(p, dict):
                continue
            try:
                pid = int(p.get("id"))
            except (TypeError, ValueError):
                continue
            if pid in idx:
                picked.append((idx[pid], p.get("why_usable", "")))
    else:
        picked = [(idx[pid], why) for pid, why in cands[:n]]

    print(f"\n「{args.text}」\n" + "─" * 68)
    for i, (r, why) in enumerate(picked, 1):
        src = f"{r[1]}·第{r[2]}章" if r[1] else f"第{r[2]}章"
        print(f"\n{i}. 「{r[4]}」〔{r[3]}〕   ← {src}")
        print(f"   做法：{r[5]}")
        if why:
            print(f"   为什么能用在你这里：{why}")
    if args.json:
        print("\n" + json.dumps(
            [{"id": r[0], "label": r[4], "why_usable": why} for r, why in picked],
            ensure_ascii=False, indent=1))


# ─────────────────────────── 命令：stats / list ───────────────────────────
def cmd_stats(args, cfg):
    if args.db:
        cfg["store"]["db"] = args.db
    conn = open_db(cfg)   # 统一走 open_db：它会自动建表（空库上 stats/list/failures/query 都不该崩）
    q = conn.execute
    books = q("SELECT COUNT(*), COALESCE(SUM(total_chars),0) FROM books").fetchone()
    nb = q("SELECT COUNT(*) FROM beats").fetchone()[0]
    print(f"库：{cfg['store']['db']}")
    print(f"书 {books[0]} 本 / 叙事动作 {nb} 条\n")
    if not nb:
        return
    print("动作分布：")
    for m, c in q("SELECT move, COUNT(*) FROM beats GROUP BY move ORDER BY 2 DESC"):
        print(f"  {m or '(空)':<6} {c:>4}  {'#' * min(c, 40)}")
    print("\n手法名出现次数（top 15）：")
    for l, c in q("""SELECT label, COUNT(*) FROM beats GROUP BY label
                     ORDER BY 2 DESC, 1 LIMIT 15"""):
        print(f"  {c:>3}× {l}")
    tot = q("SELECT COUNT(DISTINCT label) FROM beats").fetchone()[0]
    print(f"\n唯一手法名 {tot} / 总 {nb} → 重复率 {(1-tot/nb)*100:.0f}%"
          f"（自由命名下偏低是正常的，等受控手法表）")
    print("\n各书进度：")
    for bid, t in q("SELECT id, title FROM books"):
        done = q("SELECT COUNT(DISTINCT chapter_seq) FROM beats WHERE book_id=?",
                 (bid,)).fetchone()[0]
        n2 = q("SELECT COUNT(*) FROM beats WHERE book_id=?", (bid,)).fetchone()[0]
        print(f"  《{t}》 已提 {done} 章 / {n2} 条")
    nf = q("""SELECT COUNT(*) FROM runs r WHERE r.status != 'ok'
              AND (r.chapter_seq = 0 OR NOT EXISTS (SELECT 1 FROM beats k
                     WHERE k.book_id = r.book_id AND k.chapter_seq = r.chapter_seq))""").fetchone()[0]
    if nf:
        print(f"\n⚠️  {nf} 条未解决的失败/可疑记录 —— 跑 `storymine.py failures` 看详情")


def cmd_list(args, cfg):
    if args.db:
        cfg["store"]["db"] = args.db
    conn = open_db(cfg)   # 统一走 open_db：它会自动建表（空库上 stats/list/failures/query 都不该崩）
    rows = conn.execute("""SELECT move, label, COUNT(*) c, MIN(how)
                           FROM beats GROUP BY move, label ORDER BY move, c DESC""").fetchall()
    if not rows:
        print("库是空的")
        return
    cur = None
    for m, l, c, how in rows:
        if m != cur:
            print(f"\n【{m}】")
            cur = m
        print(f"  {c:>3}× 「{l}」  —— {how[:46]}")


# ─────────────────────────── 命令：delete ───────────────────────────
def parse_ranges(s):
    """'30-35' / '7' / '1-5,9' → [1,2,3,4,5,9]"""
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        if b:
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(a))
    return sorted(set(out))


def _fetch_rows(c, table, where, params):
    cur = c.execute(f"SELECT * FROM {table} WHERE {where}", params)
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def cmd_delete(args, cfg):
    if args.db:
        cfg["store"]["db"] = args.db
    if not any([args.book, args.before, args.after, args.chapters]):
        raise SystemExit("至少给一个条件：--book / --before / --after / --chapters\n"
                         "（不提供「裸删全库」这个操作。）")
    if args.chapters and not args.book:
        raise SystemExit("--chapters 必须配合 --book 用——章号只在某一本书里才有意义。")

    conn = open_db(cfg)   # 统一走 open_db：它会自动建表（空库上 stats/list/failures/query 都不该崩）
    c = conn.cursor()

    # ① 定位书
    books = []
    if args.book:
        books = c.execute("SELECT id,title,total_chars FROM books WHERE title LIKE ?",
                          (f"%{args.book}%",)).fetchall()
        if not books:
            allb = [r[0] for r in c.execute("SELECT title FROM books")]
            raise SystemExit(f"没有匹配「{args.book}」的书。库里现有：" +
                             ("、".join(allb) if allb else "（空）"))

    # ② 组装筛选条件
    bcond, bparams = [], []
    if books:
        bcond.append("book_id IN (%s)" % ",".join("?" * len(books)))
        bparams += [b[0] for b in books]
    if args.before:
        bcond.append("created_at < ?")
        bparams.append(args.before)
    if args.after:
        bcond.append("created_at > ?")
        bparams.append(args.after)
    seqs = parse_ranges(args.chapters) if args.chapters else None
    if seqs:
        bcond.append("chapter_seq IN (%s)" % ",".join("?" * len(seqs)))
        bparams += seqs
    bwhere = " AND ".join(bcond) if bcond else "1=1"

    # ③ 预览
    bcols, brows = _fetch_rows(c, "beats", bwhere, bparams)
    if not brows:
        print("没有匹配的条目，什么都没删。")
        return

    rcond, rparams = [], []
    if books:
        rcond.append("book_id IN (%s)" % ",".join("?" * len(books)))
        rparams += [b[0] for b in books]
    if args.before:
        rcond.append("ts < ?")
        rparams.append(args.before)
    if args.after:
        rcond.append("ts > ?")
        rparams.append(args.after)
    if seqs:
        rcond.append("chapter_seq IN (%s)" % ",".join("?" * len(seqs)))
        rparams += seqs
    rwhere = " AND ".join(rcond) if rcond else "0=1"
    rcols, rrows = _fetch_rows(c, "runs", rwhere, rparams) if rwhere != "0=1" else ([], [])

    cc, crows = [], []
    if seqs:
        cwhere = "book_id IN (%s) AND seq IN (%s)" % (
            ",".join("?" * len(books)), ",".join("?" * len(seqs)))
        cc, crows = _fetch_rows(c, "chapters", cwhere, [b[0] for b in books] + seqs)

    drop_books = bool(args.book) and not any([args.before, args.after, args.chapters])

    print("将删除：")
    cond_txt = []
    if args.book:
        cond_txt.append(f"书名含「{args.book}」")
    if args.before:
        cond_txt.append(f"入库时间早于 {args.before}")
    if args.after:
        cond_txt.append(f"入库时间晚于 {args.after}")
    if args.chapters:
        cond_txt.append(f"章 {args.chapters}")
    print("  条件：" + " 且 ".join(cond_txt))
    print("  " + "─" * 62)
    per_book = {}
    for r in brows:
        d = dict(zip(bcols, r))
        per_book.setdefault(d["book_id"], {"n": 0, "ch": set()})
        per_book[d["book_id"]]["n"] += 1
        per_book[d["book_id"]]["ch"].add(d["chapter_seq"])
    title_of = {b[0]: b[1] for b in books} if books else dict(
        c.execute("SELECT id,title FROM books").fetchall())
    for bid, s in sorted(per_book.items()):
        chs = sorted(s["ch"])
        rng = f"{chs[0]}-{chs[-1]}" if len(chs) > 1 else str(chs[0])
        print(f"  《{title_of.get(bid, bid)}》  章 {rng}（{len(chs)} 章）  {s['n']} 条")
    print("  " + "─" * 62)
    span = c.execute(f"SELECT MIN(created_at), MAX(created_at) FROM beats WHERE {bwhere}",
                     bparams).fetchone()
    print(f"  合计：{len(per_book)} 本书 / {sum(len(s['ch']) for s in per_book.values())} 章 / "
          f"{len(brows)} 条叙事动作 / {len(rrows)} 条运行记录")
    if drop_books:
        print(f"  （书本身也一并删除：{len(books)} 本）")
    if span[0]:
        print(f"  时间跨度：{span[0]} ~ {span[1]}")

    if not args.yes:
        print("\n  这是预览。确认无误后加 --yes 执行（执行前会自动备份）。")
        return

    # ④ 备份
    bk_dir = pathlib.Path(cfg["store"].get("backup_dir") or (ROOT / "backups"))
    bk_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bk = bk_dir / f"deleted_{stamp}.json"
    payload = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "filters": {"book": args.book, "before": args.before, "after": args.after,
                    "chapters": args.chapters},
        "beats": [dict(zip(bcols, r)) for r in brows],
        "runs": [dict(zip(rcols, r)) for r in rrows],
        "chapters": [dict(zip(cc, r)) for r in crows],
        "books": [{"id": b[0], "title": b[1]} for b in books] if drop_books else [],
    }
    bk.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    # ⑤ 执行
    c.execute(f"DELETE FROM beats WHERE {bwhere}", bparams)
    n_b = c.rowcount
    n_r = 0
    if rrows:
        c.execute(f"DELETE FROM runs WHERE {rwhere}", rparams)
        n_r = c.rowcount
    n_c = 0
    if crows:
        c.execute("DELETE FROM chapters WHERE book_id IN (%s) AND seq IN (%s)" % (
            ",".join("?" * len(books)), ",".join("?" * len(seqs))),
            [b[0] for b in books] + seqs)
        n_c = c.rowcount
    n_k = 0
    if drop_books:
        c.execute("DELETE FROM books WHERE id IN (%s)" % ",".join("?" * len(books)),
                  [b[0] for b in books])
        n_k = c.rowcount
    conn.commit()

    # ⑥ 对账
    left = c.execute("SELECT COUNT(*) FROM beats").fetchone()[0]
    print(f"\n已删除：{n_b} 条叙事动作 / {n_c} 章 / {n_r} 条运行记录 / {n_k} 本书")
    print(f"备份：{bk}（{bk.stat().st_size} bytes）")
    print(f"对账：预期删 {len(brows)} 条，实删 {n_b} 条 —— "
          + ("✓ 一致" if n_b == len(brows) else "✖ 不一致，去查"))
    print(f"库内剩余 {left} 条")
    conn.close()


def cmd_failures(args, cfg):
    """runs 表的一扇窗——失败记下来了，得看得见。"""
    if args.db:
        cfg["store"]["db"] = args.db
    conn = open_db(cfg)   # 统一走 open_db：它会自动建表（空库上 stats/list/failures/query 都不该崩）
    rows = conn.execute("""SELECT b.title, r.chapter_seq, r.status, r.note, r.ts
                           FROM runs r LEFT JOIN books b ON b.id=r.book_id
                           WHERE r.status != 'ok'
                             AND (r.chapter_seq = 0
                                  OR NOT EXISTS (SELECT 1 FROM beats k
                                                 WHERE k.book_id = r.book_id
                                                   AND k.chapter_seq = r.chapter_seq))
                           ORDER BY r.id""").fetchall()
    if not rows:
        print("没有未解决的失败/可疑记录 ✓")
        return 0
    kinds = {}
    for r in rows:
        note = str(r[3])
        real = "rejected" if ("DataInspectionFailed" in note or "inappropriate content" in note) else r[2]
        kinds[real] = kinds.get(real, 0) + 1
    print(f"未解决 {len(rows)} 条：" + "   ".join(f"{k} × {v}" for k, v in kinds.items()))
    print("（只列**当前仍未解决**的——已经重跑补上的章不再显示）\n")
    for t, ch, st, note, ts in rows:
        note = str(note)
        # 历史记录可能把"内容审查拒"错记成 badoutput —— 按事实重判（两条重跑都没用）
        real = "rejected" if ("DataInspectionFailed" in note or "inappropriate content" in note) else st
        print(f"  《{t or '?'}》 {'章' + str(ch) if ch else '整体'}  [{real}]  {ts}")
        print(f"      {note[:150]}")
    print("\n处理：")
    print("   badoutput / transient / error → **原命令重跑即可续**（已入库的章会跳过）")
    print("   rejected（渠道内容审查拒）→ 重跑没用，得换书或换渠道")
    print("   nosplit（切分可疑）→ 用 --pattern 重切，或 --skip 明确忽略")
    return len(rows)


# ─────────────────────────── 入口 ───────────────────────────
def main():
    ap = argparse.ArgumentParser(prog="storymine", description="叙事动作库 · 内核")
    ap.add_argument("--config", help="配置文件（默认项目根 config.json）")
    ap.add_argument("--db", help="覆盖库路径")
    ap.add_argument("--model", help="覆盖模型")
    ap.add_argument("--base-url", dest="base_url", help="覆盖 API 端点")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract", help="提取叙事动作入库（逐章，可续跑）")
    p.add_argument("input", help="txt 文件或目录")
    p.add_argument("--chapters", help="章范围，如 1-10（默认全部）")
    p.add_argument("--force", action="store_true", help="重跑已入库的章")
    p.add_argument("--pattern", help="自定义章标题正则（重新切分用；给了就只用它）")
    p.add_argument("--skip", action="store_true", help="切分可疑时静默跳过（忽略这一本）")
    p.add_argument("--temperature", type=float, default=None)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("query", help="卡住了就问它")
    p.add_argument("text", help="我写到……")
    p.add_argument("-n", type=int, default=None, help="返回几条（默认 5）")
    p.add_argument("--json", action="store_true", help="附 JSON")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("stats", help="库统计")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("list", help="浏览手法表")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("failures", help="列出失败/可疑的章（runs 表的一扇窗）")
    p.set_defaults(func=cmd_failures)

    p = sub.add_parser("delete", help="删除条目（默认只预览，--yes 才动手）")
    p.add_argument("--book", help="书名（模糊匹配）")
    p.add_argument("--before", help="入库时间早于（'2026-10-15' 或 '2026-10-15 21:47'）")
    p.add_argument("--after", help="入库时间晚于")
    p.add_argument("--chapters", help="章范围，如 30-35 或 1-5,9（须配合 --book）")
    p.add_argument("--yes", action="store_true", help="真执行——不加就是只看不删")
    p.set_defaults(func=cmd_delete)

    args = ap.parse_args()
    cfg = load_config(args.config)
    rc = args.func(args, cfg)
    # 退出码说真话：有章失败 / 一条没入库 → 非 0。
    # 这是唯一不需要 agent 去读输出的杠杆——只看退出码的 agent 也能收到"出事了"。
    sys.exit(1 if rc else 0)


if __name__ == "__main__":
    main()
