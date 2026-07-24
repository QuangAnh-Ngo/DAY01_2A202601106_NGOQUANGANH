# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Ở temperature 0.0, mô hình gần như trả lời giống hệt nhau mỗi lần gọi lại — một sự thật "an toàn", phổ biến, diễn đạt khô khan. Ở 0.7, câu trả lời vẫn đúng trọng tâm nhưng có sự đa dạng về cách diễn đạt và đôi khi chọn một sự thật ít quen thuộc hơn. Ở 1.2, văn phong bắt đầu lan man, thêm chi tiết không chắc chắn hoặc hơi lạc đề. Đến 1.8, câu trả lời bắt đầu mất mạch lạc rõ rệt — lặp từ, câu cú gãy, đôi khi trộn lẫn thông tin không liên quan tới Hà Nội.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Với trợ lý soạn hợp đồng pháp lý, tôi sẽ đặt temperature rất thấp (0.0–0.2) vì văn bản pháp lý cần tính nhất quán, chính xác và có thể dự đoán được — không được phép "sáng tạo" ra điều khoản hay diễn giải mơ hồ. Với trợ lý viết slogan quảng cáo, tôi sẽ đặt temperature cao hơn (0.9–1.2) vì mục tiêu là tạo ra nhiều ý tưởng đa dạng, bất ngờ, giàu hình ảnh — sự sáng tạo quan trọng hơn tính nhất quán ở đây.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> Tổng output/ngày = 20.000 người × 2 lượt × 500 token = 20.000.000 token = 20.000 × 1K token.
> - gpt-4o: 20.000 × $0.010 = **$200/ngày** (chỉ tính phần output)
> - gpt-4o-mini: 20.000 × $0.0006 = **$12/ngày**
>
> gpt-4o-mini rẻ hơn khoảng 16–17 lần. Model lớn xứng đáng với chi phí khi tác vụ đòi hỏi suy luận phức tạp, độ chính xác cao và sai sót gây hậu quả nghiêm trọng — ví dụ trợ lý phân tích hợp đồng hoặc tư vấn y tế sơ bộ. Model nhỏ là lựa chọn đúng cho các tác vụ đơn giản, khối lượng lớn như phân loại câu hỏi FAQ, gợi ý autocomplete, hay chatbot hỗ trợ khách hàng cơ bản — nơi tốc độ và chi phí quan trọng hơn chất lượng biên.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Với persona "nhà thơ", phản hồi dùng nhiều ẩn dụ, hình ảnh ví von (ví dụ ví máy học như "đứa trẻ học qua kinh nghiệm"), câu văn mềm mại, gần như không có thuật ngữ kỹ thuật hay code. Với persona "kỹ sư senior", phản hồi ngắn gọn, chính xác, có định nghĩa rõ ràng và ví dụ code minh họa. Độ dài và cấu trúc cũng khác: bản thơ thường là văn xuôi liền mạch, bản kỹ sư có thể dùng gạch đầu dòng hoặc code block. Qua đó thấy system prompt điều khiển tốt giọng văn, mức độ kỹ thuật, định dạng và trọng tâm nội dung — nhưng không đảm bảo thay đổi độ chính xác thông tin cốt lõi.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Với văn bản tiếng Việt có dấu, số token đếm bằng tiktoken thường cao hơn ước lượng `số từ / 0.75` khoảng 40–70%, vì bộ mã hóa BPE của tiktoken được huấn luyện chủ yếu trên tiếng Anh nên mỗi âm tiết tiếng Việt có dấu thường bị tách thành 2–3 token con thay vì gần 1 token như từ tiếng Anh. Nếu dùng công thức `số từ / 0.75` (vốn hiệu chỉnh cho tiếng Anh) để dự toán ngân sách API cho ứng dụng tiếng Việt, sẽ **dự toán thiếu** — chi phí thực tế cao hơn đáng kể so với ước tính, dễ dẫn tới vượt ngân sách.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Chatbot văn bản (a) hưởng lợi rõ rệt từ streaming vì người dùng nhìn thấy chữ xuất hiện ngay lập tức thay vì chờ toàn bộ phản hồi, giảm cảm giác chờ đợi (perceived latency). Trợ lý giọng nói (b) cũng hưởng lợi, dù gián tiếp hơn: hệ thống TTS có thể bắt đầu đọc câu đầu tiên ngay khi nó xuất hiện thay vì chờ toàn bộ văn bản, rút ngắn thời gian trước khi người dùng nghe được âm thanh đầu tiên. Ngược lại, pipeline dịch tài liệu chạy ngầm ban đêm (c) hầu như không cần streaming vì không có ai theo dõi trực tiếp — điều quan trọng là tổng thời gian hoàn thành và throughput xử lý hàng loạt, việc stream từng token chỉ thêm overhead xử lý mà không mang lại lợi ích trải nghiệm nào.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Với delay cố định, khi API quá tải, tất cả client thất bại gần như cùng lúc sẽ retry lại đồng loạt sau cùng một khoảng thời gian, tạo ra các đợt "retry storm" lặp lại đều đặn khiến server không kịp phục hồi. Exponential backoff giãn thời gian chờ ngày càng dài sau mỗi lần thất bại liên tiếp, giảm dần áp lực tổng thể lên server và cho hệ thống thời gian hồi phục. Tuy nhiên nếu hàng nghìn client thất bại cùng một thời điểm, chuỗi backoff của chúng vẫn đồng bộ (cùng retry ở t+delay, t+2*delay, ...), nên vẫn có các đợt tải dồn theo chu kỳ. Kỹ thuật "jitter" thêm một khoảng trễ ngẫu nhiên vào mỗi lần backoff để phá vỡ sự đồng bộ này, giúp các lượt retry trải đều theo thời gian thay vì dồn thành từng đợt, giảm hẳn hiệu ứng thundering herd.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> Persona: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt, ưu tiên ví dụ thực tế khi giải thích khái niệm."
>
> 1. Nếu xóa cụm "trả lời ngắn gọn": trợ lý có xu hướng trả lời dài dòng, giải thích lan man hơn thay vì đi thẳng vào trọng tâm.
> 2. Nếu xóa cụm "bằng tiếng Việt": trợ lý có thể chuyển sang trả lời bằng tiếng Anh hoặc trộn lẫn ngôn ngữ khi câu hỏi có từ tiếng Anh, thay vì luôn giữ tiếng Việt nhất quán.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Tình huống: ở lượt 1, người dùng nói "Mình tên Long, đang học ngành Khoa học máy tính". Sau đó cuộc trò chuyện tiếp tục sang 5–6 chủ đề khác (hỏi về temperature, về token, về streaming...). Đến lượt 7, người dùng hỏi "Bạn còn nhớ tên và ngành học của mình không?" — vì history chỉ giữ 4 lượt gần nhất, thông tin ở lượt 1 đã bị cắt bỏ nên trợ lý sẽ trả lời sai hoặc nói không biết, dù người dùng đã cung cấp thông tin đó trong cùng phiên. Cách khắc phục: thay vì cắt cứng theo số lượt, có thể trích xuất các thông tin quan trọng (tên, sở thích, ngữ cảnh cố định) thành một bản tóm tắt ngắn và luôn giữ trong system prompt bất kể history bị cắt tới đâu, hoặc định kỳ tóm tắt các lượt cũ thành 1–2 câu rồi chèn vào đầu history thay vì xóa hẳn.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
