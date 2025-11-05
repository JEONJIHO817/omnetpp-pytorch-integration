# flora 프로젝트 설정

flora 오른쪽 클릭 후, properties에서 아래 항목들 추가(+ 상단에 Configuration을 release로)

## 1. C/C++ General -> Paths and Symbols -> Includes
1) `/home/remote/libtorch/include/torch/csrc/api/include`
2) `/home/remote/libtorch/include`

*둘 다 추가해줘야 함.

## 2. 같은 곳에서 Libraries
1) `torch`
2) `torch_cpu`
3) `c10`

## 3. 마찬가지로 같은 곳에서 Libraries Paths
1) `/home/remote/libtorch/lib`

## 4. OMNeT++ -> Makemake -> src -> Options -> Link -> More -> Additional libraries
1) `torch`
2) `torch_cpu`
3) `c10`

---

이후, flora 오른쪽 클릭 -> Clean Project -> Build Project -> NetworkServerApp.cc에 아래 코드를 추가
```cpp
#include "torch/torch.h"

void NetworkServerApp::initialize(int stage)
{
    std::cout << "**********************************" << std::endl;
    torch::Tensor random_uniform_tensor = torch::rand({2, 3});
    std::cout << "Uniform Random Tensor:\n" << random_uniform_tensor << std::endl;
    ...
}
```

---

# 모델 export

터미널에서 flora 프로젝트 안에 `exported_model.py` 작성
```python
import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2) # 아주 간단히 4차원 입력에 대해 2차원으로 반환해주는 선형 모델
    
    def forward(self, x):
        return torch.relu(self.fc(x)) # 활성화 함수도 간단히 ReLU

model = TinyNet().eval()
example_input = torch.randn(1, 4)
traced = torch.jit.trace(model, example_input) # 모델을 C++에서 동작시키기 위한 API => python 코드를 저장하는게 아닌 Pytorch가 직접 예제를 가지고 연산을 수행하면서 연산 흐름을 trace에서 모델을 저장함
traced.save("example.pt")
```

---

`python3 export_model.py`로 pt파일 생성 후, omnet++ ide에서 networkserverapp.cc에 아래 코드 추가
```cpp
#include "torch/torch.h"
#include "torch/script.h"
...

void NetworkServerApp::initialize(int stage)
{
    try {
        std::cout << "Loading model..." << std::endl;
        torch::jit::Module module = torch::jit::load("/home/remote/flora/example.pt");
        torch::Tensor input = torch::rand({1, 4});
        std::cout << "Input tensor:\n" << input << std::endl;
        torch::Tensor output = module.forward({input}).toTensor();
        std::cout << "Output tensor:\n" << output << std::endl;
    }
    catch (const c10::Error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}
```
