#!/bin/bash

echo "Starting install.sh"

echo "Installing PostgreSQL"
apt-get update -y
apt-get install -y postgresql postgresql-contrib
service postgresql start
sleep 3
psql -U postgres -c 
psql -U postgres -c 
echo "PostgreSQL done"

echo "Installing MongoDB"
curl -fsSL https://www.mongodb.org/static/pgp/server-6.0.asc \
  | gpg --dearmor -o /usr/share/keyrings/mongodb-server-6.0.gpg

echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-6.0.gpg ] https://repo.mongodb.org/apt/debian bullseye/mongodb-org/6.0 main" \
  | tee /etc/apt/sources.list.d/mongodb-org-6.0.list

apt-get update -y
apt-get install -y mongodb-org
mkdir -p /data/db
mongod --dbpath /data/db --fork --logpath /tmp/mongod.log
echo "MongoDB done"

echo "install.sh complete"
