# Kịch Bản Thuyết Trình: Phân Tích Cảm Xúc Bằng LSTM & GRU Qua Mô Hình Trực Quan

Tài liệu này được thiết kế như một kịch bản/gợi ý chuyên sâu phục vụ cho việc thuyết trình về dự án Sentiment Analysis (Phân tích cảm xúc) với các kiến trúc mạng nơ-ron hồi quy tiên tiến (LSTM & GRU). Bạn có thể bám sát tài liệu này lúc chạy Web Demo để điều hướng luồng thông tin, giúp người nghe hiểu tường tận từ khái niệm khô khan đến hình ảnh hoạt động trực quan.

---

## PHẦN 1: TỔNG QUAN VÀ TIỀN XỬ LÝ (Áp dụng cho cả LSTM & GRU)

### 1.1 Đặt Vấn Đề
* **Mục tiêu:** Nhận đầu vào là một chuỗi văn bản (ví dụ: *"Thầy giảng bài rất dễ hiểu"*), mô hình sẽ hiểu trình tự của ngôn ngữ và đưa ra phán đoán cảm xúc cuối cùng (Tích cực hay Tiêu cực).
* **Vấn đề của mạng thuần (Vanilla RNN):** Bị hiện tượng triệt tiêu đạo hàm (Vanishing Gradient) không nhớ được những từ ở quá xa. Đó là lý do chúng ta cần đến LSTM và GRU.

### 1.2 Tách Từ và Nhúng Từ (Tokenization & Embedding Phase)
*(Trên Demo: Gõ câu và nhấn "Phân Tích", trang web sẽ tự động sinh thẻ từ và nhúng Vector từ trên xuống dưới).*

* **Nội dung diễn giải:**
  - Máy tính không hiểu chữ, chúng ta phải tách câu thành các **Tokens** (từng chữ hoặc âm tiết). *"Thầy"* -> *"giảng"* -> *"bài"*.
  - Sau đó, chuyển Tokens thành không gian đa chiều (ở dự án này là **Vector 128 chiều**). Bước này gọi là **Word Embedding**.
  - Từng Vector này mang hàm lượng ngữ nghĩa. Từ có nghĩa gần giống nhau sẽ nằm gần nhau trong không gian vector.
* **Cách đọc thông số trên Web:**
  - Nhấp vào bất kỳ Vector nào để mở bảng **Modal Heatmap**.
  - Heatmap thể hiện trực quan mảng 128 con số (nếu màu xanh sáng là số dương cao, màu cam/đỏ là âm). Bạn có thể chỉ ra cho người nghe thấy mảng con số này, nó giúp mạng nơ-ron thực hiện phép toán ma trận ở phía dưới.

---

## PHẦN 2: THUYẾT TRÌNH BÊN TRONG CỐI XAY CHI TIẾT - LSTM
*(Trên Demo: Ấn phím chuyển 'Bước', cho Data Vector bay chậm từ đoạn text rơi đúng vào cổng x(t)).*

### 2.1 Cấu trúc hai luồng bộ nhớ
Thay vì một luồng dữ liệu nhớ tạm bợ như RNN, LSTM (Long Short-Term Memory) quản lý 2 băng chuyền (lanes):
- **Cell State $C(t)$:** Nằm trên cùng (đường nét đứt lớn, có đồ họa như bánh răng băng chuyền). Đây là **Bộ nhớ dài hạn**, chảy xuyên suốt chuỗi mà hầu như không bị thay đổi.
- **Hidden State $h(t)$:** Nằm dưới cùng. **Bộ nhớ ngắn hạn**, tương tác nhiều với dữ liệu vào và đóng vai trò làm output cho Bước (Step) hiện tại.

### 2.2 Đọc Thông Số Trên Các Cổng (Gates)
LSTM có 4 Cổng để quyết định việc thay đổi dòng Cell State. Mỗi cổng đều sử dụng input $x(t)$ và memory từ bước trước $h(t-1)$ thông qua cơ chế chép điểm (Concat). Trỏ chuột vào các khối Gate để hiện Tooltip.

1. **Forget Gate (Cổng Quên) - Ký hiệu $f(t)$ màu Cam Đỏ**
   - **Tác dụng:** Quyết định sẽ XÓA bao nhiêu % bộ nhớ dài hạn cũ.
   - **Hàm kích hoạt:** Sigmoid $(\sigma)$. Hàm này nén giá trị trả ra từ $0$ đến $1$.
   - **Cách đọc trên UI:** Dưới chữ Forget có giá trị. Giá trị ví dụ là `0.2` (gần 0), có nghĩa cổng này đang bảo *"Hạt sạn này không liên quan, xóa 80% trí nhớ cũ đi"*. Nếu `0.9` (gần 1) tức là *"Giữ lại toàn bộ"*.

2. **Input Gate (Cổng Nạp) - Ký hiệu $i(t)$ màu Xanh Lá**
   - **Tác dụng:** Quyết định chúng ta sẽ CẬP NHẬT thông tin mới trị giá bao nhiêu %.
   - **Hàm kích hoạt:** Sigmoid $(\sigma)$ mang giá trị $0 \rightarrow 1$.

3. **Candidate (Ứng Viên) - Ký hiệu $\tilde{C}(t)$ màu Xanh Dương**
   - **Tác dụng:** Chứa *nội dung thực tế* mà từ mới mang vào. Khác với Input/Forget (chỉ đóng vai trò cấp visa/tỷ lệ), thẻ này mang nội dung dữ liệu.
   - **Hàm kích hoạt:** $\tanh$. Nén giá trị từ $-1$ đến $1$. Số âm biểu thị tính chất tiêu cực chặn nghĩa, số dương mang tính tích cực.
   - **Ý nghĩa toán học:** $i(t) \times \tilde{C}(t)$ là *"Đồng ý cho 80% (Input) của cảm xúc Tiêu cực -0.5 (Candi) đi vào trí nhớ"*.

4. **Output Gate (Cổng Xuất) - Ký hiệu $o(t)$ màu Tím**
   - **Tác dụng:** Tính toán xem sẽ trích xuất thông tin gì từ trí nhớ dài hạn $C(t)$ để chuyển cho $h(t)$ đưa lên kết quả.
   - **Hàm kích hoạt:** Sigmoid $(\sigma)$.

### 2.3 Thanh Năng Lượng (Battery Indicator) trực quan hóa Cell State
Trong Web Demo, thanh Pin nhấp nháy mô phỏng khối lượng thông tin lưu trữ bên trong **Cell State**:
* **Dấu Sét Đỏ (Drain - Rút Tải):** Lúc Forget Gate hoạt động mạnh, bạn sẽ thấy Pin tụt xuống (tương ứng với việc xóa bộ nhớ).
* **Dấu Pin Xanh (Charge - Nạp Tải):** Lúc ráp Candidate vào, thanh pin dâng % lên ứng với từ mang cảm xúc nổi trội.
* Nhờ dùng hàm Scale Sigmoid cân bằng ở giữa 50%, nếu câu mang nghĩa **Tích Cực**, pin sẽ thường xanh rực ($>70\%$). Nghĩa **Tiêu Cực** thường làm tụt pin ($<30\%$) màu Đỏ.

---

## PHẦN 3: THUYẾT TRÌNH CỐI XAY TỐI ƯU - GRU
*(Truy cập vào /gru).* 

### 3.1 Sự Thay Đổi Và Tối Ưu
GRU (Gated Recurrent Unit) sinh ra để giải quyết tốc độ tính toán chậm của LSTM. Bạn có thể nhấn mạnh:
* **Không còn Cell State:** GRU gộp **Bộ nhớ dài hạn** và **Ngắn hạn** thành một luồng duy nhất là Hidden State $h(t)$.
* **Bớt số Cổng:** Thay vì 4 khối (Forget, Input, Candidate, Output), GRU chỉ xài 3 thao tác tính toán là Reset, Update và Candi.
* **Tốc độ:** Rất tối ưu khi xử lý dữ liệu nhỏ mà chất giọng cảm xúc vẫn chính xác tương đương LSTM.

### 3.2 Luồng Hoạt Động (2 Cổng)
1. **Reset Gate (Cổng Đặt Lại - Cam) $r(t)$**
   - Lọc bỏ quá khứ. Nó quy ra một trọng số nhân trượt tiếp với $h(t-1)$ màu cam để làm giảm ảnh hưởng của quá khứ đối với *Candidate* sẽ sắp sinh ra.
   - Trên Demo, luồng nét cam sẽ rẽ ngang chui vào hộp Candi đại diện cho phép tính này.

2. **Update Gate (Cổng Cập Nhật - Tím) $z(t)$**
   - Làm nhiệm vụ của **CẢ Cổng Forget VÀ Cổng Input** của LSTM cộng lại.
   - Hàm số thực hiện lấy $(1 - z(t)) \times h(t-1)$ cộng với $z(t) \times \tilde{h}(t)$. Tức là: Chọn cân bằng xem lấy bao nhiêu % cũ và bù vảo bao nhiêu % mới.

3. **Candidate (Ứng Viên - Xanh Cyan)**
   - Tạo trạng thái gợi ý để cổng Update gate điều phối. Lưu ý ký hiệu hàm kích hoạt vẫn dùng chữ `tanh` nằm trong viên thuốc rất mượt.

---

## PHẦN 4: ĐẦU RA VÀ KẾT QUẢ DỰ ĐOÁN (Output Flow)

### 4.1 Khớp nối cuối cùng
Xuyên suốt các mô hình, qua mỗi chữ cái $(t=1, t=2,...)$, mô hình sẽ chắt lọc và truyền State đi. Khi đạt tới cell cuối cùng, $h(t_{cuối})$ sẽ chứa đựng **toàn bộ linh hồn, ngữ nghĩa của câu**.
Lúc này Hidden State được đi qua lớp mạng Dense (Linear Layer cuối) và kích hoạt bằng Softmax để hóa thành phân bố xác suất.

### 4.2 Đọc Kết Quả Suy Diễn
*(Demo sẽ tự động Auto-scroll nhanh xuống màn hình bảng điều khiển Kết Quả).*
* Lúc này thanh đếm phần trăm sẽ nhích lên dần (count-up).
* **Positive (Tích Cực) vs Negative (Tiêu Cực):** Tổng hai thanh xấp xỉ $100\%$. 
* Nhấn mạnh với người xem về những từ khóa ảnh hưởng (ví dụ: *"khó hiểu"*). Khi đi vào cell chứa từ "khó", Reset/Forget gate lập tức chặn lại và Candi bơm chỉ số tiêu cực vào bộ nhớ, dẫn đến Output ở cell cuối cho ra thanh màu đỏ báo hiệu Tiêu Cực với độ tin cậy $90\%$.

---

## THỦ THUẬT KHI THUYẾT TRÌNH BẰNG WEB DEMO NÀY

1. **Phím Tắt Kì Diệu (`?`)**:
   Nếu lỡ nhấn trượt, hãy dùng mũi tên `Trái/Phải` để next/back từng thao tác hoạt họa. Sử dụng phím `1`, `2`, `3` trên bàn phím để đổi chế độ Basic sang Advanced (hiện thông số ngầm ra ngoài SVG) rất mượt mà.
2. **Auto-Play (`Phím A`)**:
   Khi chuyển qua câu dài, hãy ấn phím `A`, sau đó thong thả cầm tách trà giải thích. Web sẽ tự động di chuyển Data vector nhảy nhót xuống, cuộn trang theo vector, và lặp vòng tròn qua từng Node một cách thần kỳ.
3. **Mẹo Hover (Trỏ Chuột)**:
   Khi đang nói về cổng nào, hãy trỏ chuột vào cổng đó. Tooltip với dòng giải thích sẽ hiện lên, giúp ban giám khảo hoặc người nghe có thể tự đọc để theo dõi mạch lập luận mà không hề ngộp. Cực kỳ uy tín!
