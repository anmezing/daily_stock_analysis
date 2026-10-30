# ===================================
# A股自选股智能分析系统 - Docker 镜像
# ===================================
# 多阶段构建：前端打包 + 后端运行

FROM node:20-slim AS web-builder

WORKDIR /app/apps/dsa-web

COPY apps/dsa-web/package.json apps/dsa-web/package-lock.json ./
RUN npm ci

COPY apps/dsa-web/ ./
RUN npm run build

# Pin to bookworm: wkhtmltopdf was removed from Debian testing (2025)
FROM python:3.11-slim-bookworm

# 设置工作目录
WORKDIR /app

# 设置时区为上海
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装系统依赖（wkhtmltopdf 含 wkhtmltoimage，用于 Markdown 转图片 Issue #289）
# Oracle/ARM 等架构上 wkhtmltopdf 可能不可用：缺失时跳过，图片渲染自动回退为文本推送。
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    fontconfig \
    libjpeg62-turbo \
    libxrender1 \
    libxext6 \
    && (apt-get install -y --no-install-recommends wkhtmltopdf \
        || echo "wkhtmltopdf unavailable on this architecture; text fallback enabled") \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY *.py ./
COPY api/ ./api/
COPY data_provider/ ./data_provider/
COPY bot/ ./bot/
COPY patch/ ./patch/
COPY src/ ./src/
COPY strategies/ ./strategies/
COPY --from=web-builder /app/static ./static/

# 创建数据目录
RUN mkdir -p /app/data /app/logs /app/reports

# 设置环境变量默认值
ENV PYTHONUNBUFFERED=1
ENV LOG_DIR=/app/logs
ENV DATABASE_PATH=/app/data/stock_analysis.db
# Web/API service
ENV WEBUI_HOST=0.0.0.0
ENV API_PORT=8000

# 暴露 API 端口
EXPOSE 8000

# 数据卷（持久化数据）
VOLUME ["/app/data", "/app/logs", "/app/reports"]

# 健康检查（FastAPI 模式）
# 注意：/api/health 现在会探测数据库，库不可用时返回 503，容器会被标记为 unhealthy。
# 不要加 "|| exit 0" 之类的兜底，否则健康检查永远成功，等于没有。
# 不提供 HTTP 服务的容器（analyzer）请在 compose 里用 healthcheck.disable=true。
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

# 默认命令（可被覆盖）
CMD ["python", "main.py", "--schedule"]
