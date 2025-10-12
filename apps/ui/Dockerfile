FROM node:20-alpine

WORKDIR /app

ARG KETABMIND_API_ORIGIN=http://127.0.0.1:8000
ARG NEXT_PUBLIC_API_URL=http://127.0.0.1:8000

ENV NODE_ENV=production
ENV KETABMIND_API_ORIGIN=${KETABMIND_API_ORIGIN}
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

COPY package.json package-lock.json ./

RUN npm ci

COPY . .

RUN npm run build

RUN addgroup -S app \
    && adduser -S app -G app \
    && chown -R app:app /app

USER app

EXPOSE 3000

CMD ["npm", "run", "start"]
