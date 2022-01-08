# git  
1. 参考教程 https://zhuanlan.zhihu.com/p/369486197
2. 常用指令
   * git status 查看状态
   * git init 初始化仓库
   * git add x.txt 将文件添加到仓库
   * git commit -m "important commit" 将修改提交到仓库
   * git reset HEAD~(数字) 取消相应数量commit
   * git log 查阅仓库日志
   * git show 显示修改
   * git restore 恢复文件
   * git branch 查看git仓库分支情况
   * git branch snd 建立新分支
   * git checkout snd 切换到分支
   * git merge snd 将该分支合并到当前分支
   * git branch -d snd 删除分支
   * git tag v1.0 为当前分支添加标签
   * git checkout v1.0 切换到相应标签
   * git push origin master 本地上传到远程仓库
   * git pull origin master 远程仓库改动下载到本地
   * git clone https://github.com/lhrrhl0419/self-study.git 克隆仓库到本地