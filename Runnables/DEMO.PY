import random
class DemoLLm:
    def __init__(self):
        print('llm created')

    def predict(self,prompt):
        response_list=[
            'Delhi is capital of india',
            'IPL is cricket league',
            'ML stands for machine learning'
        ]    

        return {'resposne':random.choice(response_list)}
    
llm=DemoLLm()
llm.predict("What is the capital of india")