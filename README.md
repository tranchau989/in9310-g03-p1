# Report

This branch holds the `(La)TeX` files for your report.

All the information required for the report should be stored in this branch.  

There is an example of a CI pipeline built in this branch as well.  See `.github/workflows/build.yml` for the setup.

## Configuration

In your own repository, you can setup different variables to change the behavior of the workflow.  In particular, you can change the defaul name of the report (`report`) that will be used to build the PDF.  The variable `FILE` corresponds to the name of the main `.tex` file and the final name of the built PDF.

Similarly, `ENGINE` defaults to `pdflatex`, but it can be changed to use any other engine to build your report.

You should go to `Settings -> Secrets and variables-> Variables` and create the intended varibles pointing to the new values.  The local variables will override the ones from the group.

## Artifacts

The result of the build will be within each workflow.  If you want to get a release, you should tag the final commit to trigger the `deploy`ment job in the workflow.

For instance,

    git tag -a v1 -m "First version of the report"

will create a tag locally, and you will need to push it to trigger the build

    git push origin v1
