"""
Vietnamese AMR Prompt Templates for Multi-Task Unified Prompt (MTUP)
Designed for consecutive task learning with explicit cues
"""

# ==============================================================================
# TEMPLATE 1: Formal Vietnamese (Học thuật)
# Best for: Academic-style training, clear structure
# ==============================================================================
MTUP_TEMPLATE_V1_FORMAL = """### NHIỆM VỤ: Phân tích ngữ nghĩa AMR hai bước

### ĐẦU VÀO:
Câu: {sentence}

### ĐẦU RA:

[BƯỚC 1: AMR_KHÔNG_BIẾN]
{amr_no_vars}

[BƯỚC 2: GÁN_BIẾN]
Quy tắc:
- Gán một biến duy nhất cho mỗi khái niệm trong AMR(không_biến).
- Tái sử dụng biến để biểu diễn đồng tham chiếu theo đúng định danh nút trong AMR(không_biến).
- Xuất AMR theo chuẩn PENMAN với biến.

AMR(có_biến):
{amr_with_vars}"""


# ==============================================================================
# TEMPLATE 2: Natural Vietnamese (Tự nhiên)
# Best for: Better model understanding, conversational style
# ==============================================================================
MTUP_TEMPLATE_V2_NATURAL = """### NHIỆM VỤ
Chuyển đổi câu tiếng Việt sang biểu diễn AMR

### CÂU ĐẦU VÀO
{sentence}

### KẾT QUẢ

## BƯỚC 1: Cấu trúc AMR không có biến
{amr_no_vars}

## BƯỚC 2: AMR hoàn chỉnh với biến

Quy tắc gán biến:
- Mỗi khái niệm → một biến duy nhất
- Khái niệm lặp lại → dùng chung biến
- Format: (biến / khái_niệm :quan_hệ ...)

AMR cuối cùng:
{amr_with_vars}"""


# ==============================================================================
# TEMPLATE 3: Instructional Vietnamese (Hướng dẫn rõ ràng)
# Best for: Strong guidance, step-by-step learning
# ==============================================================================
MTUP_TEMPLATE_V3_INSTRUCTIONAL = """Bạn là chuyên gia phân tích ngữ nghĩa tiếng Việt. Hãy thực hiện 2 bước để chuyển câu sang định dạng AMR.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 BƯỚC 1: XÂY DỰNG CẤU TRÚC AMR (KHÔNG CÓ BIẾN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Câu đầu vào:
{sentence}

Cấu trúc AMR (chỉ có khái niệm và quan hệ):
{amr_no_vars}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔗 BƯỚC 2: GÁN BIẾN VÀ TẠO ĐỒNG THAM CHIẾU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Nguyên tắc gán biến:
1. Mỗi khái niệm mới → biến mới (lấy chữ cái đầu của khái niệm)
2. Khái niệm lặp lại → dùng lại biến đã gán (thể hiện đồng tham chiếu)
3. Biến được đặt tên: chữ_cái hoặc chữ_cái + số (ví dụ: n, n2, n3, p, c, ...)

AMR hoàn chỉnh với biến:
{amr_with_vars}"""


# ==============================================================================
# TEMPLATE 4: Compact Vietnamese (Gọn nhẹ)
# Best for: Smaller models (4B), token efficiency
# ==============================================================================
MTUP_TEMPLATE_V4_COMPACT = """### Phân tích AMR hai giai đoạn

Câu: {sentence}

[Giai đoạn 1 - Cấu trúc AMR]
{amr_no_vars}

[Giai đoạn 2 - Gán biến]
Quy tắc: Khái niệm mới → biến mới. Khái niệm lặp → dùng lại biến.

{amr_with_vars}"""


# ==============================================================================
# TEMPLATE 5: Chain-of-Thought Vietnamese (Tư duy chuỗi)
# Best for: Complex reasoning, better accuracy
# ==============================================================================
MTUP_TEMPLATE_V5_COT = """### NHIỆM VỤ: Phân tích ngữ nghĩa AMR cho câu tiếng Việt

Câu cần phân tích: {sentence}

### QUÁ TRÌNH PHÂN TÍCH:

▸ Bước 1: Xác định cấu trúc ngữ nghĩa
Phân tích các khái niệm và quan hệ trong câu, tạo AMR không có biến:

<AMR_STRUCTURE>
{amr_no_vars}
</AMR_STRUCTURE>

▸ Bước 2: Gán biến và xử lý đồng tham chiếu
Quy tắc gán biến:
- Duyệt qua từng khái niệm trong AMR ở Bước 1
- Khái niệm lần đầu xuất hiện: gán biến mới (chữ cái đầu)
- Khái niệm đã gặp: sử dụng lại biến (đồng tham chiếu)
- Format cuối: (biến / khái_niệm :quan_hệ ...)

<AMR_WITH_VARIABLES>
{amr_with_vars}
</AMR_WITH_VARIABLES>"""


# ==============================================================================
# RECOMMENDED TEMPLATE FOR VIETNAMESE AMR
# Based on: Clarity, Natural flow, Token efficiency
# ==============================================================================
RECOMMENDED_TEMPLATE = MTUP_TEMPLATE_V2_NATURAL

# Template name mapping
TEMPLATE_NAMES = {
    'v1_formal': MTUP_TEMPLATE_V1_FORMAL,
    'v2_natural': MTUP_TEMPLATE_V2_NATURAL,
    'v3_instructional': MTUP_TEMPLATE_V3_INSTRUCTIONAL,
    'v4_compact': MTUP_TEMPLATE_V4_COMPACT,
    'v5_cot': MTUP_TEMPLATE_V5_COT,
    'recommended': RECOMMENDED_TEMPLATE,
}


def get_template(template_name: str = 'recommended') -> str:
    """
    Get prompt template by name

    Args:
        template_name: One of ['v1_formal', 'v2_natural', 'v3_instructional',
                               'v4_compact', 'v5_cot', 'recommended']

    Returns:
        Prompt template string
    """
    return TEMPLATE_NAMES.get(template_name, RECOMMENDED_TEMPLATE)


def format_mtup_example(
    sentence: str,
    amr_no_vars: str,
    amr_with_vars: str,
    template_name: str = 'recommended'
) -> str:
    """
    Format a training example using MTUP template

    Args:
        sentence: Vietnamese input sentence
        amr_no_vars: AMR without variables (linearized)
        amr_with_vars: AMR with variables (linearized or graph format)
        template_name: Template to use

    Returns:
        Formatted training example
    """
    template = get_template(template_name)
    return template.format(
        sentence=sentence,
        amr_no_vars=amr_no_vars,
        amr_with_vars=amr_with_vars
    )


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================
if __name__ == "__main__":
    # Example data
    sentence = "Tôi nhớ lời chủ tịch xã nhắc về vấn đề quan trọng."
    amr_no_vars = "(nhớ :pivot(tôi) :theme(lời :poss(chủ_tịch :mod(xã)) :topic(vấn_đề :mod(quan_trọng))))"
    amr_with_vars = """(n / nhớ
    :pivot(t / tôi)
    :theme(l / lời
        :poss(c / chủ_tịch
            :mod(x / xã))
        :topic(v / vấn_đề
            :mod(q / quan_trọng))))"""

    print("=" * 80)
    print("VIETNAMESE AMR PROMPT TEMPLATES - EXAMPLES")
    print("=" * 80)

    for name in ['v2_natural', 'v4_compact', 'v5_cot']:
        print(f"\n{'=' * 80}")
        print(f"TEMPLATE: {name.upper()}")
        print(f"{'=' * 80}")
        example = format_mtup_example(sentence, amr_no_vars, amr_with_vars, name)
        print(example)
        print()
