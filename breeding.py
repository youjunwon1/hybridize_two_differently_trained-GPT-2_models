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
