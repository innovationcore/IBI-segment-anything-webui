# syntax=docker/dockerfile:1

FROM nikolaik/python-nodejs:python3.11-nodejs18
WORKDIR /app
COPY .. .
RUN npm i
RUN npm install react-select
RUN npm run build

CMD ["npm", "run", "start"]
EXPOSE 3000
