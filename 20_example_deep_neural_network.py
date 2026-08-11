import torch.nn as nn
import torch
from gelu import GELU

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]),
                          GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]),
                          GELU()),
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            # 입력과 출력의 shape이 같아야 더할 수 있다
            # (마지막 층은 3차원 → 1차원으로 차원이 바뀌므로 숏컷 불가)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name}의 평균 그레이디언트는 {param.grad.abs().mean().item()}입니다.")

if __name__ == "__main__":
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1., 0., -1.]])
    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=False
    )
    print_gradients(model_without_shortcut, sample_input)

    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=True
    )
    print_gradients(model_with_shortcut, sample_input)

# 실행 결과
#
# [숏컷 없음] 출력층(layers.4)에서 입력층(layers.0)으로 역전파될수록
# 그레이디언트가 급격히 작아진다 — 그레이디언트 소실(vanishing gradient).
# layers.0.0.weight의 평균 그레이디언트는 0.00020173584925942123입니다.
# layers.1.0.weight의 평균 그레이디언트는 0.00012011158833047375입니다.
# layers.2.0.weight의 평균 그레이디언트는 0.0007152041071094573입니다.
# layers.3.0.weight의 평균 그레이디언트는 0.0013988735154271126입니다.
# layers.4.0.weight의 평균 그레이디언트는 0.005049645435065031입니다.
#
# [숏컷 있음] 모든 층의 그레이디언트가 비슷한 크기로 유지된다.
# layers.0.0.weight의 평균 그레이디언트는 0.22169791162014008입니다.
# layers.1.0.weight의 평균 그레이디언트는 0.20694106817245483입니다.
# layers.2.0.weight의 평균 그레이디언트는 0.32896992564201355입니다.
# layers.3.0.weight의 평균 그레이디언트는 0.2665732204914093입니다.
# layers.4.0.weight의 평균 그레이디언트는 1.3258540630340576입니다.
#
# 손실에 가장 가까운 layers.4의 그레이디언트가 가장 큰 것은 두 경우 모두 같다.
# 차이는 첫 번째 층: 숏컷이 없으면 0.0002까지 줄어들지만 숏컷이 있으면
# 0.22 수준으로 유지된다 — 약 1000배 차이.
# x + layer(x)를 미분하면 layer의 미분에 항상 1이 더해지므로, 그레이디언트가
# 층을 거꾸로 통과할 때마다 작은 값이 곱해져 소실되는 것을 막아준다.