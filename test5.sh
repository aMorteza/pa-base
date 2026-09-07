TOKEN=$(tr -d '\r\n' < pass2); curl -sS -H "Authorization: token $TOKEN" https://api.github.com/user
