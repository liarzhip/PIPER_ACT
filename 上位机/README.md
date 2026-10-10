# PiPER 上位机

独立上位机项目：前端使用原生 HTML、CSS、JavaScript 编写，不依赖 Vue、React 或 TypeScript；Vite 仅用于开发与构建，Python 提供采集、验证、训练和执行 API。

```text
D:\piper_act\
├── 上位机\
│   ├── frontend\          前端工程
│   │   ├── src\js\        原生 JavaScript ES modules
│   │   ├── src\css\       标准 CSS
│   │   ├── index.html     任务、采集、训练、执行
│   │   ├── review.html    回放与验证
│   │   └── dist\          npm run build 生成
│   ├── web_app.py         后端 API 与构建产物服务
│   ├── workflow.py        任务与进程管理
│   ├── tests\
│   └── requirements-web.txt
├── 数据采集\              原始采集、验证脚本
├── act\                   训练与执行脚本
├── Dataset\               数据集
└── checkpoints\           模型
```

## 安装和构建

使用已有 piper_act Python 环境，在“上位机”目录运行：

```sh
python -m pip install -r requirements-web.txt
cd frontend
npm ci
npm run build
cd ..
python web_app.py --port 5088
```

访问 http://127.0.0.1:5088 ，回放验证页面为 http://127.0.0.1:5088/review.html 。也可以直接双击上一级 `启动网页控制台.bat`，它现在指向本目录。修改前端后须重新运行 `npm run build`。

## 前端开发

终端一在“上位机”运行 `python web_app.py --port 5088`；终端二在 `frontend` 运行 `npm run dev`，访问 http://127.0.0.1:5173 。Vite 将 `/api` 请求代理到后端，保留 Host/Origin 以支持同源检查。`npm run preview` 可预览构建产物，也需要后端。

前端只调用 API；硬件访问、HDF5 和模型执行由 Python 后端处理。主页面和回放页沿用现有功能，数据及任务注册表仍位于项目根目录，迁移不会创建新的数据集副本。

## 测试

在“上位机”运行 `python -m unittest discover -s tests -v`。
详细功能说明见 `网页使用指南.md`。

## 双击启动器

上一级 `启动网页控制台.bat` 调用 `start_console.ps1`，检查环境及前端构建，等待后端健康检查通过后打开浏览器。服务在后台运行，重复双击复用已有服务；不会连接硬件或自动训练。日志保存在 `上位机/Log/console-*.log`。端口默认 5088，可传入 `-Port 5090`；`-NoBrowser` 仅启动服务。其他或旧版进程占用端口时明确报错，不会终止无关进程。
