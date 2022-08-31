cd /braindat/lab/liusl/flywire
cd sh/proxychains-ng
./configure --prefix=/usr --sysconfdir=/etc
make
sleep 60s
make install
make install-config
echo "http 192.168.16.5:3128">>/etc/proxychains.conf

