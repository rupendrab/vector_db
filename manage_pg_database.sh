# Stop	docker stop postgres-vector
# Start	docker start postgres-vector
# Restart	docker restart postgres-vector
# View logs	docker logs postgres-vector
# Delete container	docker rm -f postgres-vector

# Create a command line utility that takes in commands like start, stop, restart, logs, and delete to manage the postgres-vector container
case "$1" in
  start)
    docker start postgres-vector
    ;;
  stop)
    docker stop postgres-vector
    ;;
  restart)
    docker restart postgres-vector
    ;;
  logs)
    docker logs postgres-vector
    ;;
  delete)
    docker rm -f postgres-vector
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|logs|delete}"
    exit 1
    ;;
esac
