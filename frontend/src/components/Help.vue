<template>
  <el-card shadow="hover" class="help-card">
    <template #header>
      <div class="card-header">
        <i class="fa-solid fa-question-circle header-icon"></i>
        <span class="header-title">帮助中心</span>
      </div>
    </template>

    <div class="help-inner">
      <el-tabs v-model="activeTab" class="help-tabs">
        <el-tab-pane label="项目介绍" name="intro">
          <div class="tab-pane-body help-tab-body">
            <div class="help-section">
              <h4>产品定位</h4>
              <p>
                <strong>视频质量分析平台</strong>面向「单条短视频」做<strong>多模态</strong>质量评估：从画面、声音与弹幕文本等信号中提取特征，
                输出可解释的评分（综合分、多维雷达、按时间片的折线趋势），便于对比不同稿件或同一稿件重编码前后的质量变化。
              </p>
              <p>
                支持<strong>本地上传</strong>（多种容器格式）与<strong>B 站视频 URL 拉取</strong>；任务在用户登录后与<strong>账号绑定</strong>，列表与配置均按用户隔离。
              </p>

              <h4>技术亮点（摘要）</h4>
              <ul>
                <li>前端：<strong>Vue 3</strong>（组合式 API）+ <strong>Vite</strong> + <strong>Element Plus</strong> + <strong>ECharts</strong>；开发态通过代理访问 API，减少跨域配置成本</li>
                <li>后端：<strong>FastAPI</strong> + <strong>SQLAlchemy</strong>；默认 <strong>SQLite</strong> + WAL，适合单机交付与毕设演示</li>
                <li>分析链路：依赖 <strong>PyTorch</strong>、<strong>Transformers</strong>、<strong>MoviePy</strong> 等，实际速度与显存/CPU 及模型规模强相关</li>
                <li>鉴权：<strong>B 站扫码</strong>换取 JWT；管理员 mid 可由环境变量与设置页「管理员 UID」合并配置</li>
              </ul>

              <h4>适用场景</h4>
              <ul>
                <li>上传或引用一条视频，快速得到「能不能发、观感大概如何」的量化参考</li>
                <li>对比多次导出或转码版本（配合「重分析」与列表中的涨幅字段）</li>
                <li>教学、验收、论文实验：需要可复现的打分与片段级曲线</li>
              </ul>

              <h4>使用边界</h4>
              <ul>
                <li>评分由<strong>训练模型</strong>决定，不是主观人工审核；不同模型版本结果不可简单横向对比</li>
                <li>浏览器内<strong>播放</strong>与<strong>解码分析</strong>能力不完全相同：非 MP4/H.264 可能在网页中无法播放，但服务端仍可能完成分析</li>
                <li>若视频仅存于 <code>s3://</code>，在线播放可走预签名 URL；<strong>完整分析流水线通常仍要求本地可读文件</strong></li>
              </ul>

              <div class="help-note">
                <strong>首次部署提示：</strong>请按「部署与环境」准备 Python 依赖与 <code>backend/model/</code> 下权重；
                首次推理会加载模型，耗时可明显长于后续请求。
              </div>
            </div>
          </div>
        </el-tab-pane>

        <el-tab-pane label="功能说明" name="features">
          <div class="tab-pane-body help-tab-body">
            <div class="help-section">
              <h4>界面与导航</h4>
              <ul>
                <li><strong>首页</strong>：左侧为播放器 + 折线图；中间为数据块与两组雷达；右侧为「数据源控制」（上传 + 已上传视频列表）</li>
                <li><strong>登录</strong>：未登录时全屏二维码遮罩；成功后顶栏显示昵称，下拉含<strong>个人资料</strong>、<strong>关于我们</strong>、<strong>登出</strong></li>
                <li><strong>设置</strong>：需登录；路由守卫未带 token 时会回首页，登录后可通过顶部菜单再次进入</li>
                <li><strong>帮助</strong>：本页，可不登录浏览（部分能力说明仍依赖你已了解登录流程）</li>
              </ul>

              <h4>上传与任务类型</h4>
              <ul>
                <li>
                  <strong>单组文件选择（右侧上传区）</strong>：同一批选择中，最多 <strong>3 个视频</strong>与 <strong>3 个弹幕</strong>（合计最多 6 个文件），单文件前端限制约 <strong>2GB</strong>；
                  适合 manual 配对「视频 + 同名/同组弹幕」后一次提交
                </li>
                <li><strong>批量本地上传</strong>：仅多选<strong>视频</strong>，单次最多 <strong>10</strong> 个，由服务端为每个文件单独建任务（无共享弹幕）</li>
                <li><strong>B 站 URL</strong>：在 URL 框粘贴稿件链接并「解析」；下载落盘后再排队分析，清晰度与是否带弹幕受下载配置与稿件本身影响</li>
                <li><strong>上传进度条</strong>：反映浏览器到服务器的传输进度；传输结束后状态会进入队列/分析阶段，请以列表中的<strong>状态标签</strong>为准</li>
              </ul>

              <h4>任务状态（列表与接口一致）</h4>
              <ul>
                <li><strong>待处理（pending）</strong>：已建记录，尚未进入分析主流程或等待调度</li>
                <li><strong>待分析（downloaded）</strong>：链接下载等已完成落盘，等待正式推理（可与 pending 一起在界面上理解为「排队/未出分」类状态）</li>
                <li><strong>处理中（processing）</strong>：服务端正在解码、抽帧、推理等，请勿重复提交同一任务</li>
                <li><strong>已完成（completed）</strong>：可点「查看」加载大屏，可看折线/雷达/播放</li>
                <li><strong>失败（failed）</strong>：可看服务端返回的改进建议（若有）；满足条件时可「重分析」</li>
              </ul>

              <h4>列表与检索</h4>
              <ul>
                <li>工具条<strong>单行</strong>：<strong>状态下拉</strong> + <strong>文件名关键词</strong> + 「搜索」；回车等同点击搜索</li>
                <li>变更筛选会<strong>清空当前列表</strong>并从第一页重新请求；向下滚动触底自动加载下一页（默认每页 <strong>30</strong> 条）</li>
                <li>列表<strong>固定按上传时间降序</strong>（最新在上），界面不提供其它排序方式</li>
                <li>行内可能展示<strong>缩略图</strong>、<strong>涨幅</strong>（相对历史分数的变化，无数据时显示占位符）</li>
              </ul>

              <h4>大屏交互</h4>
              <ul>
                <li><strong>折线图</strong>：片段级得分；鼠标在曲线上移动可切换「当前片段」的总体分与雷达；点击某一竖条区域可将视频<strong>跳到对应片段起点并播放</strong></li>
                <li><strong>雷达图</strong>：视频五维（清晰、色彩、饱和度、稳定、亮度等）与音频五维（音量、音质等）</li>
                <li><strong>弹幕</strong>：需提供弹幕文件或 URL 下载成功附带弹幕；播放器旁可勾选「显示弹幕」</li>
                <li><strong>流媒体</strong>：播放 URL 带 token；支持 Range；对象存储场景下可能 302 到短期有效的预签名地址</li>
              </ul>

            <h4>设置（管理员含「管理员 UID」）</h4>
            <ul>
              <li><strong>视频处理</strong>：片段数量、每片段帧数、目标分辨率——影响算力与耗时</li>
              <li><strong>音频处理</strong>：采样率</li>
              <li><strong>敏感词</strong>：支持检索与标签列表展示。<strong>管理员</strong>可直接增删、批量保存词库、处理待审申请；<strong>普通用户</strong>仅能提交审核并在「我的申请」查看结果</li>
              <li><strong>管理员 UID</strong>（仅管理员可见）：维护库内 B 站 mid，与 <code>ADMIN_BILIBILI_MIDS</code> 合并；保存后用户<strong>下次登录</strong>按合并列表同步管理员/普通用户身份</li>
              <li><strong>下载设置</strong>：影响 URL/拉流场景下的清晰度、音轨与是否抓弹幕</li>
            </ul>
              <p class="help-muted">
                服务端还提供 <code>GET /api/sensitive-words/rules</code> 返回完整规则结构（如分类、是否正则、白名单、动作）；
                当前 Web 设置页以字面量词与审核流为主，集成系统可自行调用该接口。
              </p>

              <h4>可选基础设施</h4>
              <ul>
                <li><strong>健康检查</strong>：<code>GET /api/health</code> 返回进程存活 JSON，便于容器编排或监控</li>
              </ul>
            </div>
          </div>
        </el-tab-pane>

        <el-tab-pane label="使用指南" name="guide">
          <div class="tab-pane-body help-tab-body">
            <div class="help-section">
              <h4>第一次使用（推荐顺序）</h4>
              <ol class="help-ol">
                <li>确认后端已启动（默认 <code>http://127.0.0.1:8000</code>），前端 dev 代理或生产反代已指向该地址</li>
                <li>打开首页 → B 站 App 扫码 → 顶栏出现用户名即登录成功</li>
                <li>在右侧上传区选择视频（可附带 ass/xml 弹幕）或粘贴 B 站 URL 解析</li>
                <li>在「已上传视频」中观察状态；完成后点「查看」载入左侧大屏</li>
                <li>需要调整分析粒度或下载质量时，进入「设置」修改并「保存配置」</li>
              </ol>

              <h4>登录与账号</h4>
              <p>
                扫码成功后，浏览器会在本地保存 <strong>JWT</strong> 与简要用户信息。若 token 过期或接口返回 <strong>401</strong>，前端会清空凭证并回到首页，
                需要重新扫码。访问「设置」而未登录时，路由会重定向到首页，可在地址栏看到 <code>redirect</code> 查询参数。
              </p>

              <h4>上传操作细节</h4>
              <ul>
                <li>单组模式：建议一次选齐「某主视频 + 对应弹幕」，避免后续难以配对</li>
                <li>批量模式：仅视频、最多 10 条；适合无弹幕或弹幕稍后单独处理的批量素材</li>
                <li>超大文件可能触发浏览器或网关超时，可适当压缩分辨率与码率后再传</li>
              </ul>

              <h4>列表筛选</h4>
              <p>
                在列表上方<strong>同一行</strong>操作：先选状态（可清空表示「全部状态」），再在输入框填写文件名片段，点「搜索」或回车。
                条件变化后列表会重新加载；继续下拉可分页加载剩余条目。
              </p>

              <h4>查看与解读结果</h4>
              <ul>
                <li>仅「已完成」任务建议点「查看」；失败任务可阅读提示后尝试「重分析」或更换片源</li>
                <li>悬停折线不同位置时，中部数据块会在「整体评分 / 片段评分」语义间切换（具体文案以界面为准）</li>
                <li>若播放器黑屏但状态已完成，优先检查格式是否为浏览器友好编码（H.264 + AAC 的 MP4 最稳）</li>
              </ul>

              <h4>维护记录</h4>
              <ul>
                <li><strong>删除</strong>：从列表移除任务并清理关联上传文件（以服务端实现为准）</li>
                <li><strong>重分析</strong>：对「已完成」「失败」且路径仍可读的任务可再次入队；进行中请勿重复点</li>
              </ul>

              <h4>设置与敏感词</h4>
              <p>
                普通用户保存配置时，不会批量覆写全局敏感词表；需管理员审批通过的词才会进入词库。
                管理员在「敏感词」标签可同时维护待审队列与词库内容。
              </p>

              <div class="help-note">
                <strong>提示：</strong>修改视频片段数、分辨率等会改变后续新任务算力消耗；已有已完成任务的分数不会自动重算，除非对该任务执行「重分析」。
              </div>
            </div>
          </div>
        </el-tab-pane>

        <el-tab-pane label="常见问题" name="faq">
          <div class="tab-pane-body help-tab-body">
            <div class="help-section">
              <h4>分析与性能</h4>
              <p class="faq-q">Q: 上传后很久一直是「处理中」？</p>
              <p class="faq-a">
                A: 长视频、高分辨率、或 GPU 忙碌都会拉长耗时。若长时间无进度，可查看服务端日志是否有解码或显存错误。
              </p>
              <p class="faq-q">Q: 状态变成「失败」怎么办？</p>
              <p class="faq-a">
                A: 详情或提示里可能有 <code>suggestions</code> 建议。常见原因：文件损坏、编码极端、磁盘满、模型文件缺失。可换片源或检查服务器环境后「重分析」。
              </p>
              <p class="faq-q">Q: 折线图是平的或只有一个点？</p>
              <p class="faq-a">A: 可能当前视频没有片段级分数或仅有一条汇总；也与设置里「片段数」及实际时长有关。</p>

              <h4>格式与播放</h4>
              <p class="faq-q">Q: 支持哪些扩展名？</p>
              <p class="faq-a">
                A: 视频允许 mp4、avi、mov、wmv、flv、mkv；弹幕允许 ass 与 B 站导出的 xml。浏览器播放推荐 MP4（H.264）。
              </p>
              <p class="faq-q">Q: 列表显示已完成但播放器黑屏？</p>
              <p class="faq-a">
                A: 多为浏览器不支持该编码。可下载到本地用专业播放器验证；或将视频转码为 H.264/AAC 的 MP4 再上传。
              </p>
              <p class="faq-q">Q: 弹幕不显示？</p>
              <p class="faq-a">
                A: 确认已勾选「显示弹幕」；任务需关联弹幕文件；URL 任务依赖下载是否包含弹幕及解析是否成功。
              </p>

              <h4>列表与账号</h4>
              <p class="faq-q">Q: 筛选/搜索没有结果？</p>
              <p class="faq-a">
                A: 确认已登录；尝试清空状态或缩短关键词；列表顺序固定为上传时间降序，新任务应出现在最前（在未加筛选时）。
              </p>
              <p class="faq-q">Q: 突然退回首页并要求重新登录？</p>
              <p class="faq-a">A: Token 失效或后端 JWT 配置变更。重新扫码即可；生产环境务必设置稳定的 <code>JWT_SECRET</code>。</p>
              <p class="faq-q">Q: 如何成为管理员？</p>
              <p class="faq-a">
                A: 两种方式合并生效：（1）环境变量 <code>ADMIN_BILIBILI_MIDS</code>；（2）已是管理员的用户在设置页「<strong>管理员 UID</strong>」中写入库内列表。你的 mid 出现在<strong>合并后</strong>列表中时，<strong>下次扫码登录</strong>会同步为管理员。首名管理员通常需依赖环境变量或直接向数据库/bootstrap 写入。
              </p>

              <h4>B 站与网络</h4>
              <p class="faq-q">Q: URL 解析失败？</p>
              <p class="faq-a">
                A: 检查链接是否为有效稿件、地区限制、版权或会员专享；服务器网络需能访问 B 站；登录 Cookie（sessdata）需在有效期内。
              </p>
              <p class="faq-q">Q: 前端报「无法连接服务器」？</p>
              <p class="faq-a">
                A: 确认 uvicorn 已监听；开发时 Vite 代理目标 <code>VITE_DEV_API_TARGET</code> 是否正确；生产时 <code>VITE_BACKEND_ORIGIN</code> 是否与 API 同源策略、HTTPS 混合格式相匹配。
              </p>

              <h4>配置与敏感词</h4>
              <p class="faq-q">Q: 普通用户为何不能保存敏感词列表？</p>
              <p class="faq-a">
                A: 防止普通用户直接改全局词库。请使用「提交审核」；通过后由管理员统一入库。
              </p>
              <p class="faq-q">Q: 重置按钮会清敏感词吗？</p>
              <p class="faq-a">
                A: 设置页「重置」主要针对<strong>视频/音频/下载</strong>等默认配置逻辑；敏感词是否被改写以后端与管理员动作为准，操作前请留意提示。
              </p>

              <h4>评分含义</h4>
              <p class="faq-q">Q: 分数绝对准确吗？</p>
              <p class="faq-a">
                A: 分数来自深度学习模型回归/分类头，是相对质量度量，非人工主观打分；满分约定为 100，越高通常表示综合观感越好，但不要用于法律或合规结论。
              </p>
            </div>
          </div>
        </el-tab-pane>

        <el-tab-pane label="术语与提示" name="glossary">
          <div class="tab-pane-body help-tab-body">
            <div class="help-section">
              <h4>名词</h4>
              <ul>
                <li><strong>mid</strong>：B 站账号数字 ID，个人空间 URL 或资料页可见；用于配置管理员白名单</li>
                <li><strong>JWT</strong>：登录成功后颁发的访问令牌，请求受保护接口时在 Header 中带 <code>Authorization: Bearer …</code></li>
                <li><strong>BackgroundTasks</strong>：FastAPI 在响应返回后于同进程调度的后台任务，本项目中用于触发视频分析流水线</li>
                <li><strong>预签名 URL</strong>：对象存储生成的短期可读链接，浏览器播放器可直接访问，过期后需重新请求流接口</li>
                <li><strong>片段（clip）</strong>：将长视频按时长切分为多段，每段单独打分后在折线上显示</li>
              </ul>
              <h4>开发与调试提示</h4>
              <ul>
                <li>浏览器开发者工具 → Network：可查看 <code>/api/videos</code> 筛选参数与响应中的 <code>total</code></li>
                <li>后端日志级别为 INFO 时，可观察任务阶段与异常栈；Windows 下注意文件占用导致删除重试</li>
                <li>首次启动若下载 HuggingFace 权重失败，请配置镜像或离线放置模型到 <code>backend/model/</code> 对应路径</li>
              </ul>
              <h4>隐私与数据</h4>
              <ul>
                <li>视频与弹幕文件默认落在服务器 <code>backend/uploads</code>；数据库为 SQLite 文件，路径取决于进程工作目录</li>
                <li>生产环境请限制 API 暴露面，使用 HTTPS，并轮换 <code>JWT_SECRET</code></li>
              </ul>
            </div>
          </div>
        </el-tab-pane>

        <el-tab-pane label="部署与环境" name="deploy">
          <div class="tab-pane-body help-tab-body">
            <div class="help-section">
              <h4>环境要求（经验值）</h4>
              <ul>
                <li><strong>Python</strong>：建议 3.10+；创建 venv 后 <code>pip install -r backend/requirements.txt</code></li>
                <li><strong>Node.js</strong>：建议 18+；在 <code>frontend</code> 执行 <code>npm install</code></li>
                <li><strong>硬件</strong>：分析阶段占用 GPU/CPU 与内存显著；演示机建议独立显卡与 16GB+ 内存</li>
              </ul>

              <h4>本地开发（最常见）</h4>
              <ol class="help-ol">
                <li>仓库<strong>根目录</strong>启动后端：<code>uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000</code></li>
                <li><code>frontend</code> 目录：<code>npm run dev</code>；默认将 <code>/api</code>、<code>/uploads</code> 代理到 <code>http://127.0.0.1:8000</code>（可用 <code>VITE_DEV_API_TARGET</code> 修改）</li>
                <li>浏览器访问 Vite 提示的本地端口（常见为 5173）</li>
              </ol>

              <h4>生产构建（前端）</h4>
              <ul>
                <li>在 <code>frontend</code> 执行 <code>npm run build</code>，得到 <code>dist/</code></li>
                <li>构建前可设置 <code>VITE_BACKEND_ORIGIN=https://你的API主机</code>（无尾斜杠），使静态站点直连 API 域名</li>
                <li>由 Nginx/Caddy 等托管静态文件，并将 <code>/api</code>、<code>/uploads</code> 反代到 Uvicorn</li>
              </ul>

              <h4>后端环境变量（摘要）</h4>
              <ul>
                <li><code>ADMIN_BILIBILI_MIDS</code>：管理员 mid，逗号分隔；与设置页「管理员 UID」库内配置合并</li>
                <li><code>JWT_SECRET</code>、<code>ACCESS_TOKEN_EXPIRE_MINUTES</code></li>
                <li><code>CORS_ORIGINS</code>：逗号分隔的前端源；含 Cookie 场景慎用 <code>*</code></li>
              </ul>

              <h4>数据落盘</h4>
              <ul>
                <li><code>video_rating.db</code>：固定在项目根目录（与启动命令 cwd 无关）</li>
                <li><code>backend/uploads</code>：上传媒体与缩略图等（相对后端包路径解析）</li>
              </ul>

              <h4>启动检查清单</h4>
              <ul>
                <li><code>GET /api/health</code> 返回 <code>{"status":"ok"}</code></li>
                <li><code>GET /docs</code> 可打开 Swagger</li>
                <li>前端登录页能拉到二维码图片请求成功</li>
              </ul>
            </div>
          </div>
        </el-tab-pane>

        <el-tab-pane label="技术支持" name="support">
          <div class="tab-pane-body help-tab-body">
            <div class="help-section">
              <h4>自助排查</h4>
              <ol class="help-ol">
                <li>确认本页「部署与环境」中后端与前端均在运行</li>
                <li>用无痕窗口排除浏览器插件拦截</li>
                <li>对照「常见问题」缩小范围（网络 / 格式 / 权限 / Token）</li>
                <li>保留请求失败时的 HTTP 状态码与响应 <code>detail</code> 文案，便于反馈</li>
              </ol>

              <h4>获取帮助</h4>
              <p>
                以下为<strong>占位示例</strong>，请替换为贵校、课题组或公司实际联系方式；开源仓库可改为 Issue 与讨论区链接。
              </p>
              <ul>
                <li>邮箱：support@video-quality-analysis.com</li>
                <li>电话：400-123-4567</li>
                <li>工作时间：周一至周五 9:00–18:00</li>
              </ul>

              <div class="help-note">
                报告问题时建议附带：浏览器与版本、操作步骤、接口路径、相关截图（勿泄露真实 Cookie 或 token）。
              </div>
            </div>
          </div>
        </el-tab-pane>
      </el-tabs>
    </div>
  </el-card>
</template>

<script setup>
import { ref } from 'vue';

const activeTab = ref('intro');
</script>

<style scoped>
.help-card {
  height: 100%;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.help-card :deep(.el-card__body) {
  flex: 1;
  min-height: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  padding: 12px 16px;
}

.help-inner {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.help-tabs {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.help-tabs :deep(.el-tabs__header) {
  flex-shrink: 0;
  margin: 0 0 8px;
}

.help-tabs :deep(.el-tabs__content) {
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.help-tabs :deep(.el-tab-pane) {
  height: 100%;
  overflow: hidden;
}

.help-tab-body {
  height: 100%;
  overflow-y: auto;
  padding-right: 4px;
  box-sizing: border-box;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 18px;
  font-weight: 600;
  color: #0f172a;
}

.header-icon {
  color: #2563eb;
  font-size: 18px;
}

.help-section {
  padding: 10px 0;
}

.help-section h4 {
  margin: 18px 0 10px 0;
  color: #2563eb;
  font-size: 16px;
}

.help-section h4:first-child {
  margin-top: 0;
}

.help-section p {
  margin: 10px 0;
  line-height: 1.65;
  color: #333;
}

.help-section ul {
  margin: 10px 0 10px 20px;
}

.help-section li {
  margin: 6px 0;
  line-height: 1.55;
  color: #333;
}

.help-ol {
  margin: 10px 0 10px 22px;
  padding: 0;
  line-height: 1.6;
  color: #333;
}

.help-ol li {
  margin: 8px 0;
}

.help-section code {
  font-size: 0.9em;
  background: #f3f4f6;
  padding: 2px 6px;
  border-radius: 4px;
  color: #1f2937;
}

.help-note {
  background: linear-gradient(90deg, #eff6ff 0%, #f8fafc 100%);
  border-left: 4px solid #2563eb;
  padding: 12px 14px;
  margin: 14px 0;
  border-radius: 0 8px 8px 0;
  line-height: 1.6;
  color: #1e3a5f;
  font-size: 14px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}

.help-muted {
  font-size: 13px;
  color: #6b7280;
  line-height: 1.55;
}

.faq-q {
  font-weight: 600;
  color: #1f2937;
  margin-top: 14px;
  margin-bottom: 4px;
}

.faq-q:first-of-type {
  margin-top: 0;
}

.faq-a {
  margin-top: 0;
  margin-bottom: 8px;
  padding-left: 0;
  color: #374151;
}

.help-tab-body::-webkit-scrollbar {
  width: 6px;
}

.help-tab-body::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.help-tab-body::-webkit-scrollbar-thumb {
  background: #d1d5db;
  border-radius: 3px;
  transition: background 0.3s ease;
}

.help-tab-body::-webkit-scrollbar-thumb:hover {
  background: #9ca3af;
}
</style>
