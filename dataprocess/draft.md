请帮我完善prompt表达，只针对我回复的内容进行修改。另外请优化表达，对于冗余的内容进行精简或删除。

1. **“生成噪音变量的数量倍率”的含义**：
   - 对于每条安静数据切片，分别随机匹配`生成噪音变量的数量倍率`条噪音数据切片，将其加到安静数据中。
   - 这个倍率是一个整数，表示对于一条安静数据切片，生成倍率条加噪数据切片。例如，如果倍率为2，则对于每条安静数据切片，随机选择2条噪音数据切片进行叠加。

2. **噪音数据的添加方式**：
   - 噪音数据与安静数据是直接进行元素级别的相加
   - 在滑动窗口切片后，安静数据切片的形状是(4, 2000)，噪音数据原始的形状是(4, 5000)，需要对噪音数据进行切片或处理以匹配安静数据的形状
   - **随机性**：噪音数据切片的选择是完全随机的
   - 这一部分请在确保噪音选择随机性的前提下，提出一个你觉得高效的具体实现方案

3. **数据对齐与同步**：
   - **时间轴对齐**：噪音数据和安静数据在时间轴上需要对齐吗，还是只需形状一致即可？不需要对齐，形状一致即可
   - **通道数匹配**：`accresult`的第一维是4，代表4个通道，这在噪音和安静数据都有这四个通道，且按顺序对于，你只需要在5000维度上切片，保留原始的4个通道。

4. **输出文件的变量命名和存储**：
   - **元数据**：是否需要在输出文件中保存其他元数据信息，如原始文件名、添加的噪音信息等？添加1个元素，添加噪音的路径

5. **文件保存路径和命名**：
   - **目录创建**：程序是否需要自动创建不存在的目录？需要
   - **文件命名规则**：在`*.mat`文件的命名上，是否有特定的规则，还是沿用原始安静数据的文件名？沿用并添加合适的后缀

6. **时间戳的位置**：
   - **添加位置**：在保存`addNoiseTextCopy.JSON`时，您提到需要添加当前时间的时间戳，格式为`%Y%m%d%H%M%S`。
     - **需要澄清**：这个时间戳是添加到文件名中，例如`addNoiseTextCopy_20231113093000.JSON`，还是添加到文件内容内部？文件内部

7. **滑动窗口参数的灵活性**：
   - **参数配置**：滑动窗口的大小和步长是否需要在`addNoiseText.JSON`中配置，以便灵活调整？需要
   - **边界条件**：在进行滑动窗口切片时，如何处理无法满足窗口大小的数据段？丢弃
