TOKEN=$(tr -d '\r\n' < pass2); printf 'length=%s prefix=%s\n' "${#TOKEN}" "${TOKEN:0:4}"
