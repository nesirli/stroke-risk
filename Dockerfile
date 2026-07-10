FROM ubuntu:latest
LABEL authors="nasir"

ENTRYPOINT ["top", "-b"]