#!/bin/bash
PORT=${1:?"Usage: $0 <port>"}
lsof -ti :$PORT | xargs kill -9 2>/dev/null && echo "Stopped server on port $PORT" || echo "No server on port $PORT"
