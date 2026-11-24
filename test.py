from gtts import gTTS
from playsound import playsound

text = 'Ù tai là tiếng kêu không mong muốn có nguồn gốc từ chính hệ thống thính giác hoặc các cơ quan lân cận, thường không thể nghe được bởi người khác. Phần lớn ù tai là những tiếng kêu đơn âm, nhưng có trường hợp tiếng ù có dạng là những âm phức, như tiếng sóng biển, tiếng dế kêu, tiếng chuông reo atau tiếng hơi nước thoát qua chỗ hẹp.'

tts = gTTS(text=text, lang='vi')
tts.save('voice.mp3')

playsound('voice.mp3')