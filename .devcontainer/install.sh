#!/bin/bash
set -e

echo "Installing PostgreSQL..."
sudo apt-get update -y
sudo apt-get install -y postgresql postgresql-contrib
sudo service postgresql start
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
sudo -u postgres psql -c "CREATE DATABASE recipe_db;"
echo "✓ PostgreSQL done"

echo "Installing MongoDB..."
curl -fsSL https://www.mongodb.org/static/pgp/server-6.0.asc | \
  sudo gpg --dearmor -o /usr/share/keyrings/mongodb-server-6.0.gpg

echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-6.0.gpg ] \
https://repo.mongodb.org/apt/debian bullseye/mongodb-org/6.0 main" | \
  sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list

sudo apt-get update -y
sudo apt-get install -y mongodb-org
sudo mkdir -p /data/db
sudo chown -R $(whoami) /data/db
mongod --dbpath /data/db --fork --logpath /tmp/mongod.log
echo "✓ MongoDB done"
