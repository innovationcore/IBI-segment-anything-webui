#!/bin/sh
cd /opt/IBI-segment-anything-webui
nohup python3 scripts/server.py > backend.log &
npm run dev
