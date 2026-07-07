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
        
        print("=== DATABASE TABLE SUMMARY ===")
        db_script = (
            "docker exec -i translate_tarjomeh_1 python -c \"\n"
            "import sqlite3\n"
            "conn = sqlite3.connect('jobs/jobs.db')\n"
            "c = conn.cursor()\n"
            "for table in ['jobs', 'chunks', 'memory_state', 'job_log']:\n"
            "    try:\n"
            "        count = c.execute(f'SELECT count(*) FROM {table}').fetchone()[0]\n"
            "        print(f'Table {table} has {count} rows')\n"
            "    except Exception as e:\n"
            "        print(f'Table {table} error:', e)\n"
            "\""
        )
        status, out, err = run_ssh_command(ssh, db_script)
        print(out.encode(encoding, errors='replace').decode(encoding))
        
    finally:
        ssh.close()

if __name__ == "__main__":
    main()
