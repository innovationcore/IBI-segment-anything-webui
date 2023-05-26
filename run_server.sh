#!/bin/sh
cd /opt/IBI-segment-anything-webui
nohup python scripts/server.py > backend.log &
npm run dev
