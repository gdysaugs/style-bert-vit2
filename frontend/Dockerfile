# Use an official Node.js runtime as a parent image
# Alpine Linux is used for a smaller image size
FROM node:18-alpine

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json (or yarn.lock)
# This step benefits from Docker layer caching
COPY package*.json ./

# Install project dependencies
# If you use yarn, replace 'npm install' with 'yarn install'
RUN npm install

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on (Vite's default is 5173)
# This needs to match the port in docker-compose.yml and vite.config.ts
EXPOSE 5173

# Command to run the development server
# '--host' allows the server to be accessible from outside the container
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"] 