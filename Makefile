.PHONY: test demo check

test:
	python3 -m unittest discover -s tests -v

demo:
	python3 -m purrhaps examples/hello.prr --stats

check:
	@for source in examples/*.prr examples/projects/*/*.prr; do python3 -m purrhaps "$$source" --check || exit; done
