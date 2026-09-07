TOKEN=$(tr -d '\r\n' < pass); curl -sS -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $TOKEN" https://api.github.com/user
