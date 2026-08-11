from gelu import GELU
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

gelu, relu = GELU(), nn.ReLU()

# -3부터 3까지 같은 간격으로 나눈 100개의 입력값을 만든다.
x = torch.linspace(-3, 3, 100)

# 동일한 입력값에 GELU와 ReLU를 각각 적용해 출력값을 구한다.
y_gelu, y_relu = gelu(x), relu(x)

# 두 그래프를 나란히 담을 8×3인치 크기의 도화지를 만든다.
plt.figure(figsize=(8, 3))

# 출력값과 함수 이름을 묶어 순회한다. i는 subplot 위치를 위해 1부터 시작한다.
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    # 1행 2열로 나눈 영역 중 i번째 영역을 선택한다.
    plt.subplot(1, 2, i)
    # x를 가로축, 활성화 함수의 출력값 y를 세로축으로 그린다.
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

# 제목과 축 이름이 겹치지 않도록 subplot 사이의 여백을 자동 조정한다.
plt.tight_layout()

# 완성된 그래프를 화면에 표시한다.
plt.show()
