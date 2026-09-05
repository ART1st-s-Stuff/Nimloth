# E0110：不得用重跑替代实现审计

## 错误

人类要求研究 action-logit zero spread 是否由当前实现错误造成后，AI没有先完成
boundary hidden、LM head和传递链审计，而是把“重试上一项工作”误解为重新运行完整GPU校准。

## 后果

ID174重复消耗8×H800约20分钟，只证明zero-spread现象可复现；它本身不能证明实现正确，
也没有回答人类要求的根因问题。

## 正确做法

1. 当当前阻塞项是实现正确性时，先使用已有artifact和CPU/source审计定位根因。
2. 重跑只能验证可复现性；不得把可复现性冒充实现正确性证据。
3. 未完成审计前不得启动新的GPU calibration或canary。
