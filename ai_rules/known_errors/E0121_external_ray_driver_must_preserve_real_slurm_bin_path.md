# E0121：external Ray driver必须保留真实Slurm二进制路径

## 已发生错误
ID183 multi-node Job `520684`已成功建立2×4 Ray cluster，但driver环境把`PATH`重写为venv加`/usr/bin`。runner随后调用到`/usr/bin/scontrol` Python wrapper；固定server Python没有`colorama`，任务在output/W&B/model前失败。

## 原因
multi-node launcher只传播了Python/cache环境，没有把`module load slurm`提供的真实Slurm client路径传播进nested driver step。

## 正确做法
external Ray/Slurm launcher必须把已验证的`/cm/shared/apps/slurm/current/bin`置于nested raylet和driver的`PATH`最前，并在占GPU前验证所需二进制可执行。不得依赖`/usr/bin` wrapper，也不得用安装`colorama`掩盖错误的client选择。
