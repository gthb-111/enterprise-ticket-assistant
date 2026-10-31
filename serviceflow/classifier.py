from __future__ import annotations

from dataclasses import dataclass, field


CATEGORY_LABELS = {
    "billing": "账单与支付",
    "account": "账号与安全",
    "logistics": "物流与履约",
    "refund": "退款与售后",
    "technical": "产品与技术",
    "service": "投诉与服务",
    "general": "其他咨询",
}

TEAM_BY_CATEGORY = {
    "billing": "财务支持组",
    "account": "账号安全组",
    "logistics": "履约运营组",
    "refund": "售后服务组",
    "technical": "技术支持组",
    "service": "客户成功组",
    "general": "一线服务组",
}

CATEGORY_TERMS: dict[str, dict[str, float]] = {
    "billing": {
        "扣款": 4,
        "扣费": 4,
        "付款": 3,
        "支付": 3,
        "账单": 4,
        "发票": 5,
        "重复收费": 7,
        "重复扣费": 8,
        "重复扣款": 8,
        "多扣": 5,
        "价格": 2,
        "余额": 2,
        "充值": 3,
        "消费记录": 4,
        "自动续费": 4,
        "billing": 4,
        "invoice": 4,
        "charge": 2,
    },
    "account": {
        "登录": 3,
        "密码": 5,
        "账号": 3,
        "验证码": 5,
        "冻结": 4,
        "被盗": 6,
        "异地": 3,
        "注销": 4,
        "实名": 3,
        "手机号": 3,
        "mfa": 5,
        "password": 5,
        "account": 3,
    },
    "logistics": {
        "物流": 5,
        "快递": 5,
        "包裹": 5,
        "发货": 4,
        "签收": 4,
        "配送": 4,
        "运单": 5,
        "运输": 4,
        "到货": 3,
        "丢件": 6,
        "破损": 4,
        "delivery": 4,
        "shipping": 4,
    },
    "refund": {
        "退款": 6,
        "退回": 4,
        "退货": 5,
        "还能退": 5,
        "退吗": 3,
        "不想要": 3,
        "取消订单": 5,
        "售后": 4,
        "到账": 3,
        "原路返回": 4,
        "七天无理由": 5,
        "refund": 5,
        "return": 3,
    },
    "technical": {
        "报错": 5,
        "故障": 5,
        "闪退": 6,
        "打不开": 5,
        "登录失败": 5,
        "接口": 4,
        "系统": 3,
        "bug": 5,
        "error": 4,
        "崩溃": 5,
        "白屏": 5,
        "无法使用": 4,
        "无法操作": 5,
        "结算": 4,
        "企业版": 4,
        "加载": 3,
        "technical": 4,
    },
    "service": {
        "投诉": 6,
        "态度": 5,
        "客服": 3,
        "服务差": 6,
        "没人处理": 5,
        "12315": 7,
        "监管": 5,
        "曝光": 5,
        "起诉": 7,
        "不满意": 3,
        "反复": 2,
        "complaint": 5,
        "supervisor": 4,
    },
}

HIGH_RISK_TERMS = {
    "资金被盗": 10,
    "账号被盗": 10,
    "银行卡被盗": 10,
    "盗刷": 10,
    "遭遇诈骗": 10,
    "信息泄露": 8,
    "隐私泄露": 8,
    "起诉": 8,
    "律师函": 8,
    "12315": 7,
    "市场监管局": 7,
    "媒体曝光": 6,
    "人身安全": 10,
    "医疗事故": 10,
    "数据丢失": 7,
}

URGENT_TERMS = {
    "紧急": 4,
    "立刻": 3,
    "马上": 3,
    "现在": 2,
    "一直": 1,
    "多次": 2,
    "反复": 2,
    "已经三天": 3,
    "等了一周": 4,
    "严重影响": 4,
    "无法使用": 1,
    "丢件": 4,
    "全公司": 2,
    "影响结算": 4,
    "全部": 1,
}

NEGATIVE_TERMS = {
    "不满意": 2,
    "失望": 3,
    "生气": 3,
    "差评": 4,
    "垃圾": 3,
    "欺骗": 4,
    "没人处理": 4,
    "投诉": 4,
}


@dataclass
class Classification:
    category: str
    category_label: str
    confidence: float
    scores: dict[str, float]
    priority: str
    urgency_score: int
    risk_flags: list[str] = field(default_factory=list)
    requires_human: bool = False
    reasons: list[str] = field(default_factory=list)


def _term_score(text: str, terms: dict[str, float]) -> tuple[float, list[str]]:
    lowered = text.lower()
    score = 0.0
    matches: list[str] = []
    for term, weight in terms.items():
        if term.lower() in lowered:
            score += weight
            matches.append(term)
    return score, matches


def classify(text: str, customer_tier: str = "standard") -> Classification:
    scores: dict[str, float] = {}
    category_matches: dict[str, list[str]] = {}
    for category, terms in CATEGORY_TERMS.items():
        score, matches = _term_score(text, terms)
        scores[category] = score
        category_matches[category] = matches

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_category, top_score = ordered[0]
    second_score = ordered[1][1]
    if top_score == 0:
        top_category = "general"
        confidence = 0.38
    else:
        gap = max(0.0, top_score - second_score)
        evidence = min(top_score / 12.0, 1.0)
        confidence = 0.42 + 0.38 * evidence + 0.18 * min(gap / 8.0, 1.0)
        confidence = min(confidence, 0.97)

    risk_score, risk_flags = _term_score(text, HIGH_RISK_TERMS)
    urgent_score, urgent_matches = _term_score(text, URGENT_TERMS)
    negative_score, negative_matches = _term_score(text, NEGATIVE_TERMS)
    urgency_score = int(risk_score + urgent_score + negative_score)

    if risk_flags or urgency_score >= 7:
        priority = "P0"
    elif urgency_score >= 4 or customer_tier == "vip" and negative_score > 0:
        priority = "P1"
    elif top_category in {"billing", "account", "refund", "technical"}:
        priority = "P2"
    else:
        priority = "P3"

    reasons: list[str] = []
    if category_matches.get(top_category):
        reasons.append(
            f"命中{top_category}意图关键词："
            + "、".join(category_matches[top_category][:4])
        )
    else:
        reasons.append("未命中明确业务意图，进入通用咨询")
    if compute_score_gap(top_score, second_score) <= 2 and top_score > 0:
        reasons.append("前两名意图分数接近，置信度已降权")
    if risk_flags:
        reasons.append("命中高风险词：" + "、".join(risk_flags))
    if urgent_matches:
        reasons.append("命中时效/情绪词：" + "、".join(urgent_matches[:3]))
    if customer_tier == "vip":
        reasons.append("VIP 客户策略：提高服务优先级")

    requires_human = priority in {"P0", "P1"} or confidence < 0.56
    return Classification(
        category=top_category,
        category_label=CATEGORY_LABELS[top_category],
        confidence=round(confidence, 3),
        scores={key: round(value, 2) for key, value in scores.items()},
        priority=priority,
        urgency_score=urgency_score,
        risk_flags=risk_flags,
        requires_human=requires_human,
        reasons=reasons,
    )


def compute_score_gap(top: float, second: float) -> float:
    return max(0.0, top - second)

