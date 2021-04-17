# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

DATA = {
    'thalassemia_syn':{
        'symptom':"Các triệu chứng thường xuất hiện ở bệnh là :  \
                    Ở mức độ nhẹ bệnh ít biểu hiện ra bên ngoài người bệnh chỉ bị phát hiện khi có bệnh lí khác như nhiễm trùng, phẫu thuật, có thai....\n\
                    Ở mức độ nặng hơn là vàng da, gan, lá nách to bất thường. Cơ thể chậm phát triển, xương bị biến dạng nếu không chữa trị kịp thời sẽ rất nguy hiêm \
                    ",
        'reason':"Nguyên nhân mắc bệnh là do  gen đột biến di truyền từ cha hoặc mẹ. Vì vậy khuyến cáo mọi người lên kiểm tra sức khỏe trước hôn nhân tránh di truyền bệnh tật cho con cái sau này.",
        'concept': 'Thalassemia còn có tên gọi khác là bệnh tan máu bẩm sinh.\
                    Đây là bệnh có tính di truyền ,cơ thể người bệnh tạo ra 1\
                    lượng lớn hemoglobin khiến số lượng lớn các hồng cầu bị phá hủy quá mức dẫn đến tình trạng thiếu máu.\nĐây là bệnh cực kì nguy hiểm, gây tổn hại nặng nề tới sức khỏe và có thể khiến nạn nhân tử vong.',
                        
        'prevent':'Đây là bệnh di truyền vì vậy các biện pháp chủ yếu hướng tới định hướng hôn nhân và sinh sản:\
                    \n1.Các cặp vợ chồng chuẩn bị có thai hoặc đang mang thai, đặc biệt các gia đình đã có bệnh nhân Thalassemia nên được tư vấn và chẩn đoán tiền hôn nhân\n\
                    2.Các cặp vợ chồng trong giai đoạn thai kì cần sàng lọc phát hiện bệnh sớm cho thai nhi',
        'treatment':'Việc điều trị bệnh Thalassemia phụ thuộc vào loại và mức độ nghiêm trọng của triệu chứng liên quan. Bác sĩ sẽ đưa ra một phác đồ điều trị tốt nhất cho trường hợp cụ thể của bạn như: Truyền máu; Cấy ghép tủy xương; Sử dụng thuốc và chất bổ sung; Phẫu thuật có thể để loại bỏ lá lách hoặc túi mật.\n\
            \n  Bạn có thể tham khảo bài viết của chung tôi để hiểu rõ hơn: https://vinmec.com/vi/tin-tuc/thong-tin-suc-khoe/te-bao-goc-cong-nghe-gen/cac-phuong-phap-dieu-tri-benh-thalassemia-tan-mau-bam-sinh/.'
    },
    'ung_thư_gan_syn':{
        'symptom':"Các triệu chứng bệnh ung thư gan giai đoạn sớm thường rất khó nhận biết. Hầu hết những triệu chứng sớm của bệnh ung thư đều bị bỏ qua vì tưởng chừng chỉ là một phản ứng bình thường của cơ thể.\n\
                    Các biểu hiện phổ biến gồm: Rối loạn tiêu hóa, Đau và sưng bụng, Nước tiểu sẫm màu, Vàng da,...",
        'reason':"Ung thư gan hay gặp trên nền gan xơ, chiếm tỷ lệ đến 80%. Các nguyên nhân có thể dẫn đến xơ gan gây ung thư hóa bao gồm xơ gan do rượu, xơ gan thứ phát do nhiễm virus viêm gan B, virus viêm gan C dẫn đến ung thư tế bào gan sau 20 – 40 năm, xơ gan do nhiễm sắt. Tuy nhiên vẫn có một tỷ lệ nhiễm virus viêm gan B, C dù chưa có xơ gan vẫn bị ung thư gan.\n\
            Dùng thuốc tránh thai kéo dài cũng có thể là nguyên nhân ung thư gan. Sử dụng thuốc tránh thai trong thời gian dài tạo nên Adenoma (u tuyến) trong gan dễ tiến triển thành ung thư biểu mô tế bào gan.\n\
                Chất Aflatoxin của nấm Aspergillus có mặt trong các loại thực phẩm như lạc, đỗ bị mốc cũng có thể là nguyên nhân gây ung thư gan.",
        'concept': 'Ung thư gan, hay còn gọi là ung thư gan nguyên phát, là sự phát triển và lan rộng nhanh chóng của các tế bào không lành mạnh trong gan. Khi các tế bào gan ung thư hóa, gan không thể thực hiện chức năng thích hợp, dẫn tới các tác động có hại và nghiêm trọng đến cơ thể.',           
        'prevent':'Tiêm phòng Vacxin ngừa virus viêm gan B có thể giảm thiểu nguy cơ nhiễm virus viêm gan B gây ung thư gan. Không ăn các thực phẩm như lạc, đỗ tương đã bị mốc.',
        'treatment':'cần phát hiện và điều trị ung thư gan càng sớm càng tốt.\nMột số biện pháp điều trị bao gôm phẫu thuật cắt gan, thắt động mạch gan, ghép gan.\n Để hiểu rõ hơn từng phương pháp bạn có thể tham khảo bài viết của chúng tôi: https://www.vinmec.com/vi/tin-tuc/thong-tin-suc-khoe/cac-phuong-phap-dieu-tri-ung-thu-gan/'
    },
    'ung_thư_trực_tràng_syn':{
        'symptom':"Ung thư đại tràng thường không được chú ý vì các triệu chứng sớm nghèo nàn và ít gây sự chú ý với người bệnh:\nCác triệu chứng sau đây có thể là dấu hiệu của ung thư đại tràng:\n \
            1. Thay đổi thói quen đại tiện: xen kẽ giữa táo bón và tiêu chảy \n\
            2. Máu trong phân\n \
            3. Đau bụng quặn cơn, ậm ạch đầy hơi, bí trung tiện, các dấu hiệu của tắc ruột do u lớn làm bít tắc lòng đại tràng\n\
            4. Giảm cân, thiếu máu không biết lý do ",
        'reason':'''
            Nguyên nhân ung thư đại tràng bao gồm:\n
            Polyp đại tràng: Là nguyên nhân quan trọng gây ung thư đại tràng. Theo một nghiên cứu, trên 50% trường hợp ung thư đại tràng phát sinh trên cơ sở của polyp đại tràng. Số lượng polyp càng nhiều thì tỷ lệ ung thư hoá càng cao.\n

            Các bệnh đại tràng mãn tính: Ung thư đại tràng có thể phát sinh trên tổn thương của các bệnh: lỵ, amip, lao, giang mai, thương hàn và các bệnh lý khác của đại tràng như viêm loét đại tràng mãn tính\n
            Chế độ ăn uống ít chất bã, nhiều mỡ và đạm động vật: Chế độ ăn này làm thay đổi vi khuẩn yếm khí ở đại tràng, biến acid mật và cholesterol thành những chất gây ung thư. Đồng thời thức ăn ít bã làm giảm khối lượng phân gây táo bón, chất gây ung thư sẽ tiếp xúc với niêm mạc ruột lâu hơn và cô đặc hơn, tác động lên biểu mô của đại tràng. Các chất phân hủy của đạm như indol, seatol, piridin là những chất gây ung thư trong thực nghiệm, cũng có thể gây ung thư trên người.\n
            Yếu tố di truyền: Bệnh polyp đại tràng gia đình liên quan tới đột biến của gen APC (Adenomatous polyposis coli), chiếm 1% các ung thư đại tràng. Ngoài ra, HNPCC còn gọi là hội chứng Lynch, liên quan tới gen P53, RAS và DCC. Chiếm 5% trong số các ung thư đại trực tràng.
                ''',
        'concept':''' Ung thư đại tràng hay còn gọi là ung thư ruột già là một loại ung thư thường gặp ở Việt Nam cũng như trên thế giới.\n
                      Ung thư đại tràng là một trong những loại ung thư đường tiêu hóa có tiên lượng tốt trong trường hợp phát hiện khi bệnh còn ở giai đoạn sớm hoặc các tổn thương tiền ung thư. Nếu phát hiện muộn thì khả năng điều trị rất ít hiệu quả.''',           
        
        'prevent':'''
                    Kiểm tra đại trực tràng thường xuyên:

                    \nLà một trong những cách tốt nhất để phòng tránh ung thư, Polyp tiền ung thư thường không biểu hiện triệu chứng, có thể được tìm thấy qua nội soi đại tràng vài năm trước khi ung thư xâm lấn phát triển

                    Duy trì thói quen ăn uống lành mạnh:

                    \n   Tránh ăn nhiều thịt, dầu mỡ, thức ăn chiên nướng. Hạn chế thức uống có cồn, thuốc lá. Ăn nhiều chất xơ (rau xanh, trái cây)

                    \n Thường xuyên tập thể dục
                ''',
        'treatment':'ung_thư_trực_tràng_syn thu gan'
    }

}

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import (
    SlotSet,
    UserUtteranceReverted,
    ConversationPaused,
    EventType,
    FollowupAction,
    Restarted
)
from rasa_sdk import forms
from rasa import core
from rasa.nlu import registry


from typing import Text, List, Any, Dict

from rasa_sdk import Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict


class ValidateNameForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_name_form"


    def validate_sick_entity(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `last_name` value."""

        # If the name is super short, it might be wrong.
        print(f"Last name given = {slot_value} length = {len(slot_value)}")
        
        return {"sick_entity": slot_value}



class ActionQuestionSick(forms.FormAction):
    
    def name(self) -> Text:
        return "action_question_sick"

    def request_next_slot(
        self,
        dispatcher: "CollectingDispatcher",
        tracker: "Tracker",
        domain: "DomainDict",
    ) -> Optional[List[EventType]]:
        """Request the next slot and utter template if needed,
        else return None"""
        button  = []
        button.append(
            {"title":"thanatsamia", "payload":"thalassemia"}
        )
        button.append(
             {"title":"ung thư gan", "payload":"ung thư gan"}
        )
        button.append(
             {"title":"ung thư trực tràng", "payload":"ung thư trực tràng"}
        )
        for slot in self.required_slots(tracker):
            if self._should_request_slot(tracker, slot):
                logger.debug(f"Request next slot '{slot}'")
                dispatcher.utter_message("bạn muốn hỏi về bệnh nào nhỉ ?", buttons=button, **tracker.slots)
                # dispatcher.utter_message(template=f"utter_ask_{slot}", **tracker.slots)
                return [SlotSet(REQUESTED_SLOT, slot)]

        # no more required slots to fill
        return None

    


class ActionConceptHandle(Action):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        # self.step_loop_ = 0
    def name(self) -> Text:
        return "action_concept"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        entities = [i for i in tracker.latest_message.get("entities",[]) ]
        slots = tracker.get_slot("sick_entity")
        num = tracker.get_slot("")
        if len(entities) or slots:
            if slots:slot = slots
            if len(entities):slot = entities[0]['entity']
            for i in DATA[slot]['concept'].split("\n"):
                dispatcher.utter_message(i)
            
            return [SlotSet("sick_entity",slot), SlotSet("type_slot","concept")]

        
        return []


class ActionFlowSick(Action):
    def name(self):return "action_flow_sick"
    def run(
        self, dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ):

        entities = [i for i in tracker.latest_message.get("entities",[]) if i['entity'] in ['thalassemia_syn','ung_thư_gan_syn','ung_thư_trực_tràng_syn']]
        slots = tracker.get_slot("sick_entity")
        if len(entities) or slots:
            if slots:slot = slots
            if len(entities):slot = entities[0]['entity']

            return [SlotSet("sick_entity",slot)]
        return []


class ActionReason(Action):

    def name(self) -> Text:
        return "action_reason"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        entities = [i for i in tracker.latest_message.get("entities",[]) ]
        slots = tracker.get_slot("sick_entity")
        if len(entities) or slots:
            if slots:slot = slots
            if len(entities):slot = entities[0]['entity']
            for i in DATA[slot]['reason'].split("\n"):
                dispatcher.utter_message(i)
            return [SlotSet("sick_entity",slot),  SlotSet("type_slot","reason")]
        # dispatcher.utter_message("bạn muốn hỏi về bệnh nào nhỉ ?")
        return []


class ActionSympton(Action):

    def name(self) -> Text:
        return "action_symptom"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        entities = [i for i in tracker.latest_message.get("entities",[]) ]
        slots = tracker.get_slot("sick_entity")
        if len(entities) or slots:
            if slots:slot = slots
            if len(entities):slot = entities[0]['entity']
            for i in DATA[slot]['symptom'].split("\n"):
                dispatcher.utter_message(i)
            return [SlotSet("sick_entity",slot), SlotSet("type_slot","symptom")]
        # dispatcher.utter_message(F"{DATA[slot]['symptom']}")
        return []

class ActionPrevent(Action):

    def name(self) -> Text:
        return "action_prevent"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        entities = [i for i in tracker.latest_message.get("entities",[]) ]
        slots = tracker.get_slot("sick_entity")
        if len(entities) or slots:
            if slots:slot = slots
            if len(entities):slot = entities[0]['entity']
            for i in DATA[slot]['prevent'].split("\n"):
                dispatcher.utter_message(i)
            return [SlotSet("sick_entity",slot),SlotSet("type_slot","prevent")]
        # dispatcher.utter_message("bạn muốn hỏi về bệnh nào nhỉ ?")
        return []

class ActionTreatment(Action):

    def name(self) -> Text:
        return "action_treatment"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        entities = [i for i in tracker.latest_message.get("entities",[]) ]
        slots = tracker.get_slot("sick_entity")
        if len(entities) or slots:
            if slots:slot = slots
            if len(entities):slot = entities[0]['entity']
            for i in DATA[slot]['treatment'].split("\n"):
                dispatcher.utter_message(i)
            return [SlotSet("sick_entity",slot), SlotSet("type_slot","treatment")]
        # dispatcher.utter_message("bạn muốn hỏi về bệnh nào nhỉ ?")
        return []


class ActionReversed(Action):
    def name(self):return "action_reversed"
    def run(self, dispatcher,tracker, domain):
        entities = [i for i in tracker.latest_message.get("entities",[]) ]
        slots = tracker.get_slot("sick_entity")
        intent = tracker.get_slot("type_slot")
        slot = None
        if len(entities) or slots:
            if slots:slot = slots
            if len(entities):slot = entities[0]['entity']

        if slot and intent:
            for i in DATA[slot][intent].split("\n"):

                dispatcher.utter_message(i)
            # dispatcher.utter_message("trả lời câu hỏi  về {} cho bệnh {}".format(intent,slot))

        else:
            dispatcher.utter_message("Xin lỗi mình không hiểu ý bạn !!!")
        if slot:
            return [SlotSet("sick_entity",slot)]
        return []
        

