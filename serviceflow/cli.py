from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

from .config import Settings
from .server import create_server


DEMO_TICKETS = (
    (
        "陈女士",
        "vip",
        "上周订单被重复扣款两次，已经两天了还没有退回，请马上处理。",
    ),
    (
        "王先生",
        "standard",
        "包裹显示运输中五天没有更新，物流到底什么时候能到？",
    ),
    (
        "李同学",
        "standard",
        "退款审核通过后多久到账？原路退回到银行卡。",
    ),
    (
        "赵女士",
        "standard",
        "手机号换了，验证码收不到，账号登录不上。",
    ),
    (
        "周先生",
        "enterprise",
        "后台一直报错 500，企业端所有成员都无法使用，影响今天的结算。",
    ),
    (
        "吴女士",
        "standard",
        "我已经联系客服三次都没人回复，非常失望，再不处理就投诉。",
    ),
    (
        "郑先生",
        "standard",
        "怎么申请开电子发票？",
    ),
    (
        "孙女士",
        "standard",
        "怀疑账号被盗，有人用我的账号下单，银行卡也显示异常消费。",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="serviceflow",
        description="Enterprise ticket triage and knowledge-base assistant",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=["serve", "seed-demo", "assist"],
    )
    parser.add_argument("--message", default="")
    parser.add_argument("--name", default="匿名客户")
    parser.add_argument("--tier", default="standard")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.host:
        settings = Settings(**{**settings.__dict__, "host": args.host})
    if args.port:
        settings = Settings(**{**settings.__dict__, "port": args.port})

    if args.command == "serve":
        server = create_server(settings)
        print(f"ServiceFlow AI: http://{settings.host}:{settings.port}")
        print("Press Ctrl+C to stop.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return

    from .engine import TriageEngine
    from .knowledge import KnowledgeBase
    from .storage import TicketStore

    kb = KnowledgeBase.load(settings.knowledge_dir)
    store = TicketStore(settings.db_path)
    engine = TriageEngine(kb, store, settings)
    if args.command == "assist":
        if not args.message:
            parser.error("assist requires --message")
        result = engine.assist(args.message, args.name, args.tier)
        print(json.dumps(
            {
                "ticket": result.ticket,
                "decision": result.decision,
                "explanation": result.explanation,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return

    if store.metrics()["total"] == 0:
        for name, tier, message in DEMO_TICKETS:
            engine.assist(message, name, tier)
        print(f"Seeded {len(DEMO_TICKETS)} demo tickets into {settings.db_path}")
    else:
        print("Database already contains tickets; demo seed skipped.")

