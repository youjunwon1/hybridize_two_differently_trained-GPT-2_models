# hybridize_two_differently_trained-GPT-2_models , The Possibility of Model Breeding for GPT-2:

Enhancing Model Performance: By breeding two models with different strengths, we can create a new model with better overall performance.
Expanding Model Diversity: Breeding models trained with various datasets and hyperparameters can produce new models with unique features and capabilities.
Model Exploration and Optimization: Through breeding, we can explore various model variations and find the optimal model structure.
What if we bred dogs to only produce well-behaved offspring, or to only produce those sensitive to errors?

The current problem with AI is that it tends to boast about generating something even when it's incorrect. However, if among various trained AIs, one emerges that tends to acknowledge its mistakes, we could breed that model. Over time, this could result in an AI that predominantly speaks correctly.

We can't prevent AI from making inferences based on the given data because that's what it's designed to do. But if there were an AI that could say, "I made this inference, but it's not correct according to a certain standard," we could use this AI as a parent to create offspring.

With this simple idea, among hundreds of offspring, we might find those that adhere better to these standards. By randomly distributing 50% of the genes of an already educated parent and continually breeding offspring that make inferences but don't violate these standards, we could potentially evolve AIs with a special bias toward correctness.

This method would involve evolving AI by continuously breeding descendants that are born with special weights, ensuring they adhere more strictly to established norms.

By translating the concept of selective breeding into AI development, we might foster models that not only infer based on given data but also adhere to correctness standards more rigorously.


GPT-2 모델 교배의 가능성:

모델 성능 향상: 서로 다른 강점을 가진 두 모델을 교배하여 더 나은 성능을 가진 새로운 모델을 만들 수 있습니다.

모델 다양성 확장: 다양한 학습 데이터와 하이퍼파라미터를 사용하여 훈련된 모델들을 교배하여 새로운 기능과 특성을 가진 모델을 만들 수 있습니다.

모델 탐색 및 최적화: 교배를 통해 다양한 모델 변형을 탐색하고 최적의 모델 구조를 찾을 수 있습니다.



강아지를 브리딩해서  착한 아이만  계속 브리딩해나가거나 
오류에 민감한 아이만 브리딩 해나간다면   어떨까  ?  

지금 ai의 문제는 자신이 틀려도 뭔가를 만들어 냈다는 것을 떠벌리고 싶어하는 녀석이란 말이야 
그거를  여러 트레이닝을 거친  ai들 중에서  그래도 그나마 틀림을 인정하는 모델이 나오면  
그 녀석을 브리딩해서  점점 옳은 것만 말하는 녀석을 만들 수 있지 않을까 ? 

인공지능이  주어진 자료로 추론해서 말하는 것을 막을 수 는 없다  그것이 맞으니까  

단지 주어진 자료로 추론은 하되  기준이 되는 규범이란게 있어서 추론은 했지만 규범에 어긋나기에  
이건 아니다 라고 말하는 ai가 있다면 

그 ai를 부모로 해서 자식을 만든다면 
수백의 자식들 중에서  더 규범을 잘 지키는 녀석이 나타나지 않을까 하는 단순한 생각이다  

이미 교육 받은 부모의 유전자 50%를 랜덤하게 나눠주면서  추론은 하지만 규범에 어긋나지 않는 녀석 

특별한 가중치를 갖고 태어나는 자손들을 계속 번식시키는 방법으로  진화시키면 어떨까 싶은 생각이다  



# breeding an AI trained to always critique with a general AI.





import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer


# 두 개의 사전 훈련된 GPT-2 모델을 로드합니다.
model_name_1 = 'gpt2'
model_name_2 = 'gpt2-medium'

model_1 = GPT2LMHeadModel.from_pretrained(model_name_1)

model_2 = GPT2LMHeadModel.from_pretrained(model_name_2)

# 유전 알고리즘을 위한 하이퍼파라미터 설정
num_generations = 10
population_size = 10
mutation_rate = 0.1

def crossover(model_a, model_b):
    """ 두 모델의 가중치를 교배합니다. """
    new_model = GPT2LMHeadModel(model_a.config)
    new_state_dict = {}

    for key in model_a.state_dict().keys():
        if key in model_b.state_dict():
            # 임의의 가중치 섞기 (50% 확률로 모델 A 또는 B의 가중치 선택)
            new_state_dict[key] = torch.where(torch.rand(model_a.state_dict()[key].size()) > 0.5,
                                              model_a.state_dict()[key],
                                              model_b.state_dict()[key])
        else:
            new_state_dict[key] = model_a.state_dict()[key]
    
    new_model.load_state_dict(new_state_dict)
    return new_model

def mutate(model, mutation_rate):

    """ 모델 가중치에 작은 변화를 줍니다. """
    
    for param in model.parameters():
        if torch.rand(1).item() < mutation_rate:
            noise = torch.randn(param.size()) * 0.02
            param.data.add_(noise)
    return model

# 초기 모델 풀 생성
population = [GPT2LMHeadModel.from_pretrained(model_name_1) for _ in range(population_size)]

for generation in range(num_generations):

    # 평가 및 선택 (여기서는 단순히 첫 번째 모델로 설정)
    
    selected_models = population[:2]  # 예시로 첫 두 모델을 선택

    # 교배 및 변이
    
    new_population = []
    for i in range(population_size):
        parent_a = selected_models[i % 2]
        parent_b = selected_models[(i + 1) % 2]
        child = crossover(parent_a, parent_b)
        child = mutate(child, mutation_rate)
        new_population.append(child)

    population = new_population

# 최종 모델 선택 (여기서는 첫 번째 모델로 설정)
final_model = population[0]
tokenizer = GPT2Tokenizer.from_pretrained(model_name_1)

# 모델 사용 예제
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = final_model.generate(inputs['input_ids'], max_length=50)
print(tokenizer.decode(outputs[0]))


