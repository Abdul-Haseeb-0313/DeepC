# Simple DeepC Makefile
CC = gcc
CFLAGS = -std=c99 -Wall -O2
TARGET = libdeepc.a

SRC = src/matrix.c src/layers.c src/models.c src/losses.c src/optimizers.c src/data_processing.c
OBJ = $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -Iinclude -c $< -o $@

install:
	cp include/*.h /usr/local/include/
	cp $(TARGET) /usr/local/lib/

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all install clean