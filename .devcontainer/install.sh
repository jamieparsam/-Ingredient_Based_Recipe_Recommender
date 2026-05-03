#!/bin/bash

echo "Starting install.sh"

echo "apt-get update"
sudo apt-get update -y
echo "✓ apt-get update done"

echo "Installing PostgreSQL"
sudo apt-get install -y postgresql postgresql-contrib
echo "✓ PostgreSQL installed"

echo "Starting PostgreSQL"
sudo service postgresql start
sleep 2
echo "✓ PostgreSQL started"

echo "Configuring PostgreSQL"
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';" || echo "WARN: password set failed"
sudo -u postgres psql -c "CREATE DATABASE recipe_db;" || echo "WARN: db create failed"
echo "✓ PostgreSQL configured"

echo "Adding MongoDB GPG key"
curl -fsSL https://www.mongodb.org/static/pgp/server-6.0.asc \
  | sudo gpg --dearmor -o /usr/share/keyrings/mongodb-server-6.0.gpg
echo "✓ GPG key added"

echo "Adding MongoDB repo"
echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-6.0.gpg ] https://repo.mongodb.org/apt/debian bullseye/mongodb-org/6.0 main" \
  | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
echo "✓ Repo added"

echo "apt-get update (with MongoDB repo)"
sudo apt-get update -y
echo "✓ apt-get update done"

echo "Installing MongoDB"
sudo apt-get install -y mongodb-org
echo "✓ MongoDB installed"

echo "Starting MongoDB"
sudo mkdir -p /data/db
sudo chown -R $(whoami) /data/db
mongod --dbpath /data/db --fork --logpath /tmp/mongod.log
echo "✓ MongoDB started"

echo "install.sh complete"
