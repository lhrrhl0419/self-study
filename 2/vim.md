# vim
1. 下载位置 https://www.vim.org/download.php  
2. 自带教程比较清晰易懂  
3. vim由于不需要鼠标操作，方便进行指令操作而比较好用
4. vs有对应插件
5. 常用指令
    1. 移动
        * hjkl 左下上右单字符移动
        * 0 行首
        * 数字+w/e 向后移动相应数量单词
        * (数字) G 跳转到指定行
        * gg 跳转到第一行
        * G 跳转到最后一行
    2. 写
        * x 删除光标位置单字符
        * r/R 修改单个/多个字符
        * i 在光标位置前添加文本
        * a 在光标位置后添加文本
        * A 在行末添加文本
        * o/O 在下/上方新建行并进入插入模式
        * <<或>> 调整缩进
        * d 删除 c 删除并进入插入模式以修改
            * ...w 从当前位置删除到下一单词起始
            * ...e 从当前位置删除到该单词末尾
            * ...数字w/e 从当前位置开始删除相应数量单词
            * ...$ 从当前位置删除到行末
        * (数字)dd 删除(数字)行
    3. 块操作
        * v 选择块
        * y 复制
        * vy均支持后接2w等指定选取范围
        * p 将最近删除内容/复制内容粘贴到光标后
    4. 查找与替换 
        * /(string) 查找字符串
        * n/N 下一个/上一个符合要求的字符串
        * CTRL-O/I 回退跳转/恢复跳转
        * % 跳转至与当前括号配对括号 
        * :s/(old)/(new) 替换该行第一个字符串
        * .../g 替换全行字符串
        * .../c 在替换时提示是否替换
        * :(num1),(num2)s/... 替换指定行范围字符串
        * :%s/... 替换全文件字符串
    5. 文件操作
        * :q! 丢弃修改退出
        * :w (文件名) 保存
        * 选中块后按 : 后显示 :'<,'> 后输入w (文件名) 以保存选中内容
        * :wq 保存修改退出
        * :r (文件名/指令) 将相应文本置入
    6. 其他
        * u 撤销最后执行的命令
        * U 恢复本行初始状态
        * CTRL-R 重做被撤销命令
        * CTRL-G 查看光标位置，文件信息
        * :! 执行外部命令
        * :set 设置
        * ESC 退回到正常模式
