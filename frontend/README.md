# Альфа-Новости: фронтенд

React 19, TypeScript, Vite, React Router, TanStack Query, CSS Modules.

## Запуск

```bash
nvm use            # Node 22
npm install
cp .env.example .env
npm run dev
```

Запросы на `/api/*` проксируются на `VITE_PROXY_TARGET`. По умолчанию это стенд
https://alfa-news.mooo.com, где nginx сам снимает префикс `/api`.

Для локального бэкенда (`uvicorn` или Docker Compose на порту 8000):

```env
VITE_PROXY_TARGET=http://localhost:8000
VITE_PROXY_STRIP_API=1
```

## Скрипты

| Команда | Действие |
| --- | --- |
| `npm run dev` | dev-сервер |
| `npm run build` | проверка типов и production-сборка |
| `npm run lint` | oxlint |
| `npm run format` / `format:check` | prettier |
| `npm run typecheck` | tsc |
| `npm test` | vitest |

## Структура

```text
src/
  api/          клиент, типы и хуки react-query по ресурсам (компании, новости, аналитика, чат)
  auth/         вход, регистрация, хранение токена
  components/   UI-кит, иконки, потоковый вывод текста
  layout/       каркас приложения, навигация, выбор компании
  pages/        страницы: сводка, разбор события, источники, Agent Trace, аналитика банка
  state/        выбранная компания и сохранённые события
```

Потоковые ответы бэкенда (оценки ленты и подробный разбор) читаются через SSE в `src/api`.
