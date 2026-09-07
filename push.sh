PASSWORD=$(<pass)

expect -c '
spawn git push
expect "Username:"
send "aMorteza\r"
expect "Password:"
send "'"$PASSWORD"'\r"
expect eof
'
