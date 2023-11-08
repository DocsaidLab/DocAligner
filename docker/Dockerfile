# syntax=docker/dockerfile:experimental
FROM docsaid_training_base_image:latest

WORKDIR /code

# 設置入口點脚本路徑
ENV ENTRYPOINT_SCRIPT=/entrypoint.sh

# 安裝 gosu
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# 創建入口點腳本
RUN printf '#!/bin/bash\n\
    if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then\n\
    groupadd -g "$GROUP_ID" -o usergroup\n\
    useradd --shell /bin/bash -u "$USER_ID" -g "$GROUP_ID" -o -c "" -m user\n\
    export HOME=/home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /code\n\
    exec gosu "$USER_ID":"$GROUP_ID" "$@"\n\
    else\n\
    exec "$@"\n\
    fi' > "$ENTRYPOINT_SCRIPT"

# 賦予權限
RUN chmod +x "$ENTRYPOINT_SCRIPT"

# 入口點
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]

CMD ["bash"]
