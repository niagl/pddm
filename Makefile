.PHONY: build run

BUILD_ARGUMENTS=
RUN_ARGUMENTS=
ifdef http_proxy
	BUILD_ARGUMENTS+=--build-arg http_proxy=$(http_proxy)
	RUN_ARGUMENTS+=--env http_proxy=$(http_proxy)
endif

ifdef https_proxy
	BUILD_ARGUMENTS+=--build-arg https_proxy=$(https_proxy)
	RUN_ARGUMENTS+=--env https_proxy=$(https_proxy)
endif

CONTEXT ?= $(shell pwd)

# use nvidia-docker if it is available
DOCKER := $(shell command -v nvidia-docker 2> /dev/null)
ifndef DOCKER
	DOCKER = docker
endif

clean:
	@rm -f .*.swp .*.swo
	@rm -f *.pyc

build:
	${DOCKER} build --no-cache -f=${CONTEXT}/Dockerfile -t=${IMAGE} ${BUILD_ARGUMENTS} ${CONTEXT} --build-arg USER=$(USER)

push:
	${DOCKER} push ${IMAGE}

run-cache:
	${DOCKER} run ${RUN_ARGUMENTS} --rm -it ${CACHE_IMAGE} /bin/bash

run: build
	${DOCKER} run ${RUN_ARGUMENTS} --rm -it ${IMAGE}

shell: build
	${DOCKER} run ${RUN_ARGUMENTS} --rm -it ${IMAGE} /bin/bash
