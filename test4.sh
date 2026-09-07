TOKEN=$(tr -d '\r\n' < pass2); curl -sS -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $TOKEN" https://api.github.com/user
