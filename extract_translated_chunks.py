import sys
import paramiko

def run_ssh_command(ssh, cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    exit_status = stdout.channel.recv_exit_status()
    out = stdout.read().decode('utf-8', errors='ignore')
    err = stderr.read().decode('utf-8', errors='ignore')
    return exit_status, out, err

def main():
    host = "138.124.26.69"
    user = "root"
    secret = "F4F7cmJW9uiA"
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, username=user, password=secret, timeout=30)
        encoding = sys.stdout.encoding or 'ascii'
        
        # Query completed chunks from DB
        print("=== COMPLETED TRANSLATED CHUNKS ===")
        db_script = (
            "docker exec -i translate_tarjomeh_1 python -c \"\n"
            "import sqlite3\n"
            "conn = sqlite3.connect('jobs/jobs.db')\n"
            "c = conn.cursor()\n"
            "rows = c.execute('SELECT chunk_index, text, translation, status FROM chunks ORDER BY chunk_index ASC').fetchall()\n"
            "if not rows:\n"
            "    print('No chunks found in database.')\n"
            "for r in rows:\n"
            "    print(f'\\n--- Chunk {r[0]} (Status: {r[3]}) ---')\n"
            "    print('[SOURCE TEXT PREVIEW]:', r[1][:300] + '...' if r[1] else 'None')\n"
            "    print('[TRANSLATION TEXT PREVIEW]:', r[2][:300] + '...' if r[2] else 'None')\n"
            "\""
        )
        status, out, err = run_ssh_command(ssh, db_script)
        print(out.encode(encoding, errors='replace').decode(encoding))
        
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
