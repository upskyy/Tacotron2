# Tacotron 2 (without wavenet)
 Pytorch implementation of [**Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions.**](https://arxiv.org/abs/1712.05884)  
   
 
- **Tacotron 2 system architecture**   
![image](https://user-images.githubusercontent.com/54731898/124512375-261b4b80-de13-11eb-8e7d-6d380d6208db.png)  

## Installation
```   
pip install -e .   
```   
  
## Usage
```python
from tacotron2 import Tacotron2
import torch

batch = 3
seq_length = 100
n_mel = 80
num_vocabs = 10

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

model = Tacotron2(num_vocabs=num_vocabs).to(device)
print(model)

inputs = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                           [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                           [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
input_lengths = torch.LongTensor([9, 8, 7]).to(device)
target = torch.FloatTensor(batch, seq_length, n_mel).to(device)

outputs = model(inputs, input_lengths, target)
```

## Reference
- [Tacotron2: Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884)  
- [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2)  
- [sooftware/tacotron2](https://github.com/sooftware/tacotron2)  
  
## License
```
Copyright 2021 Sangchun Ha.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```  
