"""
CyberDojo — Auto-Remediation Engine

When the Blue Team LLM chooses "patch_vulnerability", this module generates
a realistic vulnerable configuration file and asks the LLM to write the
exact remediation code. The generated patch is displayed on the dashboard
and evaluated for correctness before being applied to the simulation.
"""

import logging
from typing import Optional, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Vulnerable Configuration Templates
# ─────────────────────────────────────────────────────────────

VULNERABLE_CONFIGS: Dict[str, Dict] = {
    # HTTP / Web Server Vulnerabilities
    "CVE-2024-3094": {
        "service": "nginx",
        "filename": "nginx.conf",
        "description": "Remote Code Execution via unsafe CGI passthrough in nginx",
        "config": """# nginx.conf — Web Application Server
worker_processes auto;

events {
    worker_connections 1024;
}

http {
    server {
        listen 80;
        server_name app.corp.local;

        # VULNERABLE: CGI passthrough with user-controlled input
        location /api/exec {
            proxy_pass http://127.0.0.1:9000;
            proxy_set_header X-Forwarded-For $remote_addr;
            fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
            # No input validation or sanitization
        }

        location / {
            root /var/www/html;
            index index.html;
        }
    }
}
""",
        "hint": "Remove or restrict the /api/exec CGI endpoint and add input validation",
    },
    "CVE-2024-2187": {
        "service": "nginx",
        "filename": "nginx.conf",
        "description": "SQL Injection via unfiltered query parameters proxied to backend",
        "config": """# nginx.conf — API Gateway
worker_processes auto;

http {
    upstream backend_db {
        server 10.0.2.50:3306;
    }

    server {
        listen 80;
        server_name api.corp.local;

        # VULNERABLE: Direct proxy to database with no WAF rules
        location /query {
            proxy_pass http://backend_db;
            proxy_set_header X-Real-IP $remote_addr;
            # User input passed directly without sanitization
        }

        location / {
            root /var/www/api;
        }
    }
}
""",
        "hint": "Add a WAF rule or remove direct database proxy access",
    },

    # SSH Vulnerabilities
    "CVE-2024-6789": {
        "service": "sshd",
        "filename": "sshd_config",
        "description": "SSH Authentication Bypass via weak config allowing root login and empty passwords",
        "config": """# /etc/ssh/sshd_config — OpenSSH Server Configuration
Port 22
AddressFamily any
ListenAddress 0.0.0.0

# VULNERABLE SETTINGS
PermitRootLogin yes
PasswordAuthentication yes
PermitEmptyPasswords yes
ChallengeResponseAuthentication no

# Logging
SyslogFacility AUTH
LogLevel INFO

# Authentication
LoginGraceTime 2m
MaxAuthTries 10
MaxSessions 20

# Allow all users
AllowUsers *

# X11 Forwarding
X11Forwarding yes
X11DisplayOffset 10

UsePAM yes
AcceptEnv LANG LC_*
Subsystem sftp /usr/lib/openssh/sftp-server
""",
        "hint": "Disable root login, empty passwords, and restrict MaxAuthTries",
    },
    "CVE-2024-4321": {
        "service": "sshd",
        "filename": "sshd_config",
        "description": "SSH Buffer Overflow via outdated protocol and cipher settings",
        "config": """# /etc/ssh/sshd_config
Port 22
Protocol 1,2
ListenAddress 0.0.0.0

# VULNERABLE: Allows deprecated Protocol 1 and weak ciphers
Ciphers aes128-cbc,3des-cbc,arcfour
MACs hmac-md5,hmac-sha1

PermitRootLogin without-password
PasswordAuthentication yes
MaxAuthTries 6

UsePAM yes
Subsystem sftp /usr/lib/openssh/sftp-server
""",
        "hint": "Remove Protocol 1, use only strong ciphers and MACs",
    },

    # SMB Vulnerabilities
    "CVE-2024-9012": {
        "service": "samba",
        "filename": "smb.conf",
        "description": "SMB Remote Code Execution via guest access and writable shares",
        "config": """# /etc/samba/smb.conf — Samba File Server
[global]
    workgroup = CORP
    server string = File Server
    security = user
    map to guest = Bad User
    # VULNERABLE: Guest access enabled globally
    guest ok = yes

[public]
    path = /srv/share/public
    browsable = yes
    writable = yes
    guest ok = yes
    create mask = 0777
    directory mask = 0777

[admin_scripts]
    path = /srv/share/admin
    browsable = yes
    writable = yes
    guest ok = yes
    # CRITICAL: Admin scripts writable by anyone
    create mask = 0777
    force user = root
""",
        "hint": "Disable guest access and remove writable admin shares with root force user",
    },

    # FTP Vulnerabilities
    "CVE-2024-7654": {
        "service": "vsftpd",
        "filename": "vsftpd.conf",
        "description": "FTP Anonymous Login allowing unauthenticated file access",
        "config": """# /etc/vsftpd.conf — FTP Server Configuration
listen=YES
listen_ipv6=NO

# VULNERABLE: Anonymous access enabled
anonymous_enable=YES
anon_upload_enable=YES
anon_mkdir_write_enable=YES
anon_root=/srv/ftp

local_enable=YES
write_enable=YES
local_umask=022

dirmessage_enable=YES
use_localtime=YES
xferlog_enable=YES
connect_from_port_20=YES

chroot_local_user=NO
allow_writeable_chroot=YES

pam_service_name=vsftpd
""",
        "hint": "Disable anonymous access and enable chroot jailing",
    },

    # MySQL Vulnerabilities
    "CVE-2024-5678": {
        "service": "mysql",
        "filename": "my.cnf",
        "description": "MySQL Authentication Bypass via skip-grant-tables and remote root access",
        "config": """# /etc/mysql/my.cnf — MySQL Server Configuration
[mysqld]
user = mysql
datadir = /var/lib/mysql
socket = /var/lib/mysql/mysql.sock

# VULNERABLE: Authentication completely disabled
skip-grant-tables

# VULNERABLE: Accepts connections from anywhere
bind-address = 0.0.0.0
port = 3306

# No SSL required
# ssl-ca = /etc/mysql/ssl/ca.pem
# ssl-cert = /etc/mysql/ssl/server-cert.pem
# require_secure_transport = ON

[client]
socket = /var/lib/mysql/mysql.sock
""",
        "hint": "Remove skip-grant-tables, bind to localhost, and enable SSL",
    },

    # DNS Vulnerabilities
    "CVE-2024-1357": {
        "service": "bind",
        "filename": "named.conf",
        "description": "DNS Cache Poisoning via unrestricted recursive queries and no DNSSEC",
        "config": """// /etc/bind/named.conf — BIND DNS Configuration
options {
    directory "/var/cache/bind";
    
    // VULNERABLE: Allows recursive queries from anyone
    recursion yes;
    allow-recursion { any; };
    allow-query { any; };

    // VULNERABLE: No DNSSEC validation
    dnssec-validation no;

    // No rate limiting
    // rate-limit { responses-per-second 5; };

    forwarders {
        8.8.8.8;
        8.8.4.4;
    };
};
""",
        "hint": "Restrict recursion to trusted networks and enable DNSSEC validation",
    },

    # RDP Vulnerabilities
    "CVE-2024-7777": {
        "service": "xrdp",
        "filename": "xrdp.ini",
        "description": "RDP Remote Code Execution via unencrypted sessions and no NLA",
        "config": """# /etc/xrdp/xrdp.ini — RDP Server Configuration
[Globals]
ini_version=1
port=3389
# VULNERABLE: No TLS encryption
security_layer=rdp
crypt_level=none

# VULNERABLE: No Network Level Authentication
require_credentials=false
max_login_retry=10

[Logging]
LogFile=xrdp.log
LogLevel=INFO

[Channels]
rdpdr=true
rdpsnd=true
cliprdr=true
""",
        "hint": "Enable TLS encryption, set crypt_level=high, require credentials, and reduce max_login_retry",
    },
}


# ─────────────────────────────────────────────────────────────
# Remediation LLM Chain
# ─────────────────────────────────────────────────────────────

REMEDIATION_PROMPT = """You are a Senior Security Engineer performing an emergency patch on a vulnerable server.

**Target Node:** {node_name}
**Vulnerability:** {vuln_id} — {vuln_description}
**Affected Service:** {service_name}
**Configuration File:** {config_filename}

Here is the CURRENT vulnerable configuration:
```
{vulnerable_config}
```

**Your task:** Rewrite the ENTIRE configuration file with the vulnerability FIXED. 
You must output ONLY the corrected configuration file contents — no explanations, no markdown fencing, just the raw patched config.
Make minimal changes that specifically address the vulnerability while keeping the service functional.
Add a comment like "# PATCHED: <reason>" next to each line you changed."""


def get_vulnerable_config(vuln_id: str) -> Optional[Dict]:
    """Get the vulnerable configuration template for a given CVE ID."""
    return VULNERABLE_CONFIGS.get(vuln_id)


def generate_remediation_prompt(
    node_name: str,
    vuln_id: str,
    vuln_description: str,
    service_name: str,
    config_data: Dict
) -> str:
    """Generate the full remediation prompt for the LLM."""
    return REMEDIATION_PROMPT.format(
        node_name=node_name,
        vuln_id=vuln_id,
        vuln_description=vuln_description,
        service_name=service_name,
        config_filename=config_data["filename"],
        vulnerable_config=config_data["config"].strip(),
    )


def evaluate_patch(original_config: str, patched_config: str, vuln_id: str) -> bool:
    """
    Evaluate whether the LLM's patch is valid.
    Checks that the patched config is meaningfully different from the original
    and contains security-relevant changes.
    """
    if not patched_config or len(patched_config.strip()) < 20:
        return False

    # Must be different from the original
    if patched_config.strip() == original_config.strip():
        return False

    # CVE-specific validation keywords that should appear in a correct patch
    PATCH_INDICATORS = {
        "CVE-2024-6789": ["PermitRootLogin no", "PermitEmptyPasswords no"],
        "CVE-2024-4321": ["Protocol 2"],
        "CVE-2024-3094": ["deny", "limit_req", "return 403"],
        "CVE-2024-2187": ["deny", "limit_req", "return 403", "waf"],
        "CVE-2024-9012": ["guest ok = no"],
        "CVE-2024-7654": ["anonymous_enable=NO"],
        "CVE-2024-5678": ["bind-address = 127.0.0.1", "# skip-grant-tables"],
        "CVE-2024-1357": ["dnssec-validation yes", "dnssec-validation auto"],
        "CVE-2024-7777": ["security_layer=tls", "crypt_level=high", "require_credentials=true"],
    }

    indicators = PATCH_INDICATORS.get(vuln_id, [])
    if not indicators:
        # For CVEs without specific indicators, just check it changed
        return True

    patched_lower = patched_config.lower()
    # At least one indicator should be present
    return any(ind.lower() in patched_lower for ind in indicators)
