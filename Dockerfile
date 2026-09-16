FROM node:24.19.0-alpine AS styles
WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci
COPY templates ./templates
COPY static/css/source.css ./static/css/source.css
COPY scripts/copy-fonts.mjs ./scripts/copy-fonts.mjs
RUN npm run build:css

FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY --from=styles /build/static/css/app.css ./static/css/app.css
COPY --from=styles /build/static/fonts ./static/fonts
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
