# talks
Repository of public talks

Creating a new Presentation:
  - Create a new branch, make a copy of `template`, and `cd` there
  - git submodule current version of reveal.js
    ```bash
    git submodule add https://github.com/AsianTaco/reveal.js
    cd reveal.js
    git submodule update --init --recursive
    cd ..
    ```
  - Back in the new talk directory, copy `package.json` and `gulpfile.js` from  the **reveal.js/** folder to the talk directory via
    ```bash
    cp reveal.js/gulpfile.js reveal.js/package.json .
    ```
  - Update/install npm modules to make sure it  works with this version:
    ```
    $ npm install
    ```
  - Start the server:
    ```
    $ npm start
    ```

