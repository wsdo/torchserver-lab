## TorchServe
```
git clone https://github.com/pytorch/serve.git
```

## 安装依赖

# CUDA为可选项
```
python ./ts_scripts/install_dependencies.py --cuda=cu117
```

### 2.3、安装TorchServe
```
pip install torchserve torch-model-archiver torch-workflow-archiver
```

### 
```
python load_and_save.py --model_name_or_path /root/chatglm2-6b/ --quantization_bit 4 --cache_dir ./model
```
### 打包模型

```
torch-model-archiver --model-name chatglm2 \
    --version 1.0 \
    --serialized-file pytorch_model.bin \
    --handler ../chatglm_handler.py \
    --extra-files "config.json,configuration_chatglm.py,generation_config.json,modeling_chatglm.py,quantization.py,special_tokens_map.json,tokenization_chatglm.py,tokenizer_config.json,tokenizer.model"
```




# 处理模型
python load_and_save.py --model_name_or_path /root/chatglm2-6b --quantization_bit 4 --cache_dir ./model


# 创建保存mar文件的目录
mkdir model_store

# 进入模型文件的目录
cd model

# 打包模型文件
torch-model-archiver --model-name chatglm2  --version 1.0  --serialized-file pytorch_model.bin  --handler ../chatglm_handler.py  --export-path ../model_store  --extra-files "config.json,configuration_chatglm.py,generation_config.json,modeling_chatglm.py,quantization.py,special_tokens_map.json,tokenization_chatglm.py,tokenizer_config.json,tokenizer.model"

# 回到程序目录
cd ..

# 启动服务
torchserve --start --model-store model_store --models chatglm2.mar --ts-config config.properties



### 参数说明
```
<ul>
    <li>--model-name:  模型名称，导出后的模型文件是“模型名称.mar”</li>
    <li>--serialized-file: 模型序列化文件，这里有两种文件数据: <ul>
        <li>一种叫eager模式，包含状态字典的.pt/.pth/.bin文件；</li>
        <li>另一种是TorchScript条件下的可执行模块。</li>
    </ul>
<li>--model-file: （可选） 模型结构框架，通常只包含一个类，类是torch.nn.modules的子类。</li>
<li>--handler:  torchserver 的入口程序<b>（见下文详解）</b>。</li>
<li>--extra-files: 额外文件，模型运行等其它额外文件都可以，可以包含多个，用逗号将不同文件拼接成一个字符串。</li>
<ul><li><b>如果是huggingface的模型，一般这里需要放入模型加载所需的所有文件。</b></li></ul>
<li>--run-time: （可选）选择运行的python版本。</li>
<li>--archive-format: （可选）选择压缩文件的格式, {tgz,no-archive,default} 。可以是tgz压缩文件，也可以是mar文件。</li>
<li>--export-path: （可选）mar存档文件的保存地址，未设置则保存在当前目录。</li>
<li>-f:  强制覆盖。</li>
<li>-v --version: 模型的版本</li>
<li>-r, --requirement-f:（可选）模型环境相关的依赖包requirements.txt。</li>
</ul>
```


介绍

什么是TorchServe？
TorchServe的主要功能和优势。
安装与配置

如何安装TorchServe。
配置要求和建议。
模型的打包与部署

如何将PyTorch模型转换为TorchServe可用的格式。
使用torch-model-archiver工具。
启动和停止TorchServe服务。
模型的管理

注册模型。
列出已注册的模型。
设置模型的默认版本。
删除模型。
推理

如何发送推理请求。
同步和异步推理。
批量推理。