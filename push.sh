USERNAME="aMorteza"
PASSWORD=$(<pass)

cat > /tmp/git-askpass.sh <<EOF
#!/bin/sh
case "\$1" in
  *Username*) printf '%s\n' "$USERNAME" ;;
  *Password*) printf '%s\n' "$PASSWORD" ;;
esac
EOF

chmod 700 /tmp/git-askpass.sh
GIT_ASKPASS=/tmp/git-askpass.sh GIT_TERMINAL_PROMPT=0 git push
rm -f /tmp/git-askpass.sh
